import asyncio

import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import  CommandStart, StateFilter
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter

from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message,
    CallbackQuery,
    FSInputFile
)
import time

import config

from bot.states import UserStates
from bot.keyboards import get_keyboard
from bot.texts import *

from gpt_answer_api_id import GPTGenApiId
from oeсd_api_id import OEСDApiId
from gpt_answer_api_document import GPTGenApiDoc
from oeсd_api_document import OEСDApiDoc

TOKEN = config.bot_token

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
bot = Bot(TOKEN)

bot_logger = logging.getLogger('bot_logger')

if bot_logger.hasHandlers():
    bot_logger.handlers.clear()

# Настраиваем формат и уровень логирования только для нашего логгера
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s - %(message)s')
handler.setFormatter(formatter)
bot_logger.addHandler(handler)
bot_logger.setLevel(logging.INFO)

async def safe_delete_messages(chat_id: int, message_ids: list):
    """Функция безопасного удаления сообщений с обработкой ошибок."""
    if not message_ids:
        return

    try:
        await bot.delete_messages(chat_id=chat_id, message_ids=message_ids)
    except TelegramBadRequest:
        bot_logger.warning(f"Сообщения {message_ids} уже удалены или не существуют.")
    except TelegramForbiddenError:
        bot_logger.warning(f"Нет прав на удаление сообщений в чате {chat_id}.")
    except TelegramRetryAfter as e:
        bot_logger.warning(f"Превышен лимит запросов. Ожидание {e.retry_after} секунд...")
        await asyncio.sleep(e.retry_after)
        await safe_delete_messages(chat_id, message_ids)

@dp.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    bot_logger.info(f'Начало итерации: {curr_time}')
    data = await state.get_data()
    try:
        message_id = data["delete_messege"]
        user_data = await state.get_data() 
        await safe_delete_messages(message.chat.id, user_data.get("delete_messege", []))
    except KeyError:
        await message.answer(hello_message_text)
    await message.answer(start_message_text)
    message_excel = await message.answer(id_message_text)
    user_id = message.from_user.id
    await state.update_data(delete_messege=[message_excel.message_id], user_id=user_id)
    await state.set_state(UserStates.get_id)

@dp.message(UserStates.get_id, F.text.regexp(r'^\d+$') | (F.content_type == 'document'))
async def get_id_handler(message: Message, state: FSMContext):
    user_data = await state.get_data()
    if isinstance(message.text, str):
        bot_logger.info('Начало работы с id')
        await safe_delete_messages(message.chat.id, user_data.get("delete_messege", []))
        await state.update_data(author_id = message.text, source_status='status_id')
    else:
        bot_logger.info('Начало работы с документом')
        await safe_delete_messages(message.chat.id, user_data.get("delete_messege", []))
        user_id = user_data["user_id"]

        file_id = message.document.file_id
        file_name = f"{str(user_id)}_{message.document.file_name}"
        await state.update_data(file_name=file_name, source_status='status_document')

        file = await bot.get_file(file_id)
        file_path = file.file_path
        bot_logger.info(f'Документ "{file_name}" получен от пользователя {user_id}')
        await bot.download_file(file_path, f"files/{file_name}")

    mode_message_id = await message.answer(mode_message, reply_markup=get_keyboard('mode_kb'))
    await state.update_data(delete_messege=[mode_message_id.message_id])
    await state.set_state(UserStates.get_answer)


@dp.callback_query(UserStates.get_answer, F.data.in_([button_gpt_gen, button_oesd_gen]))
async def get_mode_handler(callback: CallbackQuery, state: FSMContext):
    bot_logger.info('выбор режима')
    user_data = await state.get_data()
    source_status = user_data['source_status']
    if source_status == 'status_id':
        if callback.data == button_gpt_gen:
            api=GPTGenApiId()
            bot_logger.info(f'Режим {button_gpt_gen}')
        elif callback.data == button_oesd_gen:
            api=OEСDApiId()
            bot_logger.info(f'Режим {button_oesd_gen}')
    elif source_status == 'status_document':
        if callback.data == button_gpt_gen:
            api=GPTGenApiDoc()
            bot_logger.info(f'Режим {button_gpt_gen}')
        elif callback.data == button_oesd_gen:
            api=OEСDApiDoc()
            bot_logger.info(f'Режим {button_oesd_gen}')

    bot_logger.info('Начало c API')
    await safe_delete_messages(callback.message.chat.id, user_data.get("delete_messege", []))
    user_id = user_data['user_id']
    author_id = int(user_data.get('author_id', 1))
    file_name = user_data.get('file_name', None)


    waiting_message_id = await callback.message.answer(waiting_message)
    try:
        if source_status == 'status_id':
            answer = api.get_answer(author_id, user_id)
        elif source_status == 'status_document':
            answer = api.get_answer(f"files/{file_name}")

        await safe_delete_messages(callback.message.chat.id, [waiting_message_id.message_id])
        
        await callback.message.answer_document(FSInputFile(answer))
        message_after = await callback.message.answer(
            "Что бы запусть бота заново напишите /start"
        )
        bot_logger.info('Конец итерации')
        await state.update_data(delete_messege=[message_after.message_id])
        await state.clear()
    except TypeError as e:
        # await state.update_data(delete_messege=[message_after.message_id])
        await callback.message.answer('Что-то пошло не по плану(')
        bot_logger.error(e)


@dp.message(UserStates.get_id, ~F.text.regexp(r'^\d+$') | (F.context_type != "document"))
async def warning_not_id(message: Message, state: FSMContext):
    data = await state.get_data()
    message_id = data["delete_messege"]
    message_id.append(message.message_id - 1)

    await safe_delete_messages(message.chat.id, data.get("delete_messege", []))
    answer_text = f"{warning_pdf_message}\n\n{id_message_text}"
    await message.answer(text=answer_text)
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id, messege_id + 1])
    data = await state.get_data()


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))