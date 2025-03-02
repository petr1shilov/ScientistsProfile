from aiogram.fsm.state import State, StatesGroup

class UserStates(StatesGroup):
    get_id = State()
    get_mode = State()
    get_answer = State()