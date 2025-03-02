#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import logging
import xlsxwriter

from bot.texts import *

import config

api_logger = logging.getLogger("api_logger")
if api_logger.hasHandlers():
    api_logger.handlers.clear()

# Настраиваем формат и уровень логирования только для нашего логгера
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s - %(message)s")
handler.setFormatter(formatter)
api_logger.addHandler(handler)
api_logger.setLevel(logging.INFO)


class OEСDApiDoc:
    def __init__(self, api_key=config.api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        oecd_df = pd.read_excel('kody_OECD.xlsx')
        self.oecd_df = oecd_df.applymap(lambda s: s.replace('\n', ' ') if isinstance(s, str) else s)

    def get_gpt_answer(self, system_prompt, user_query):
        completion = self.client.chat.completions.create(
            model= self.model,
            messages= [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )
        return completion.choices[0].message.content
    
    def categorize_scientific_fields(self, annotation_text):
        classification_result = {}

        primary_categories = self.oecd_df['1 уровень'].dropna().unique()
        api_logger.info(f'\t\tуровень 1\n')

        primary_classification = self.get_gpt_answer(
            system_prompt=(
                        "Определи, к каким областям науки из перечисленных относится данная аннотация научной статьи. "
                        "Не придумывай области. И пиши их как в примерах"
                        "Перечисли только области через запятую. "
                        "Примеры областей: \n"
                        '\n'.join(primary_categories)
                    ),
            user_query=annotation_text
        )
        primary_classification = primary_classification.split(', ')

        api_logger.info(f'\t\tуровень 2\n')
        for primary_field in primary_classification:
            secondary_categories = self.oecd_df[self.oecd_df['1 уровень'] == primary_field]['2 уровень Русское наименование'].dropna().unique()

            secondary_classification = self.get_gpt_answer(
                system_prompt=(
                        "Определи, к каким областям науки из перечисленных относится данная аннотация научной статьи. "
                        "Не придумывай области. И пиши их как в примерах"
                        "Перечисли только области через запятую. "
                        "Примеры областей: \n"
                        '\n'.join(secondary_categories)
                    ),
                user_query=annotation_text
            )
            secondary_classification = secondary_classification.split(', ')
            classification_result[primary_field] = {subfield: [] for subfield in secondary_classification}

        api_logger.info(f'\t\tуровень 3\n')
        for primary_field in classification_result:
            for secondary_field in classification_result[primary_field]:
                tertiary_categories = self.oecd_df[self.oecd_df['2 уровень Русское наименование'] == secondary_field]['3 уровень Русское наименование'].dropna().unique().tolist()
                tertiary_classification = self.get_gpt_answer(
                    system_prompt=(
                        "Определи, к каким областям науки из перечисленных относится данная аннотация научной статьи. "
                        "Не придумывай области, используй только из примера. И пиши их как в примерах"
                        "Перечисли только области через запятую. "
                        "Примеры областей: \n"
                        '\n'.join(tertiary_categories)
                    ),
                    user_query=annotation_text
                )
                tertiary_classification = tertiary_classification.split(', ')
                classification_result[primary_field][secondary_field].extend(tertiary_classification)

        return classification_result
    
    def get_oecd(self, answer):
        df_updated = pd.DataFrame([
            [outer_key, inner_key, value]
            for outer_key, inner_dict in answer.items()
            for inner_key, values in inner_dict.items()
            for value in values
        ], columns=["уровень 1", "уровень 2", "уровень 3"])

        code_list = []
        for i in range(len(df_updated)):
            try:
                wos_code = self.oecd_df[self.oecd_df['3 уровень Русское наименование'].str.lower() == df_updated.loc[i]['уровень 3'].lower()][['Коды OECD', 'Коды WoS']]
                code_list.append(str(wos_code.values.tolist()[0][0]) + ' ' + str(wos_code.values.tolist()[0][1]))
            except:
                code_list.append('----')

        df_updated['OECD Код'] = code_list

        return df_updated

    
    def dict_to_excel(self, path):
        try:
            df = pd.read_excel(path)
        except:
            df = pd.DataFrame()
            print('read df error')

        total_df = pd.DataFrame()

        for lab_num in range(len(df)):
            annotation_columns = [col for col in df.columns if 'аннотация' in col.lower()]
            api_logger.info(f'{df.loc[lab_num]["Название лаборатории / центра"]}\n')
            for annotation_num, annotation_col in enumerate(annotation_columns):
                annotation_num += 1
                api_logger.info(f'\t{annotation_col}')

                # Получаем данные по OECD
                oecd_answer = self.get_oecd(self.categorize_scientific_fields(df.loc[lab_num][annotation_col]))

                # Создаём данные для DataFrame
                name = [df.loc[lab_num]['Название структуры, к которой относится лаборатория']] * len(oecd_answer)
                lab_name = [df.loc[lab_num]['Название лаборатории / центра']] * len(oecd_answer)
                annot_num = [annotation_num] * len(oecd_answer)

                # Формируем корректный DataFrame
                answer_df = pd.DataFrame(
                    data=list(zip(name, lab_name, annot_num)), 
                    columns=['Название структуры', 'Название лаборатории', 'Номер аннотации']
                )

                # Объединяем с oecd_answer
                answer_df = pd.concat([answer_df, oecd_answer], axis=1)

                # Выводим результат
                total_df = pd.concat([total_df, answer_df], axis=0)

        total_df = total_df.reset_index(drop=True)
        path = f"{path[:-4]}_total.xlsx"
        total_df.to_excel(path, index=False, engine='xlsxwriter')
        api_logger.info("Ответ запакован в xlsx")
        return path

    def get_answer(self, file_path):
        api_logger.info("Начало работы api")
        answer_df = self.dict_to_excel(file_path)
        api_logger.info("Файл готов к отправке")
        return answer_df
