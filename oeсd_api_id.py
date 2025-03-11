#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import logging
import xlsxwriter
import requests

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


class OEСDApiId:
    def __init__(self, api_key=config.api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        oecd_df = pd.read_excel('kody_OECD.xlsx')
        self.oecd_df = oecd_df.applymap(lambda s: s.replace('\n', ' ') if isinstance(s, str) else s)


    def get_author_papers(aelf, author_id, top_n_papers=2, n_search_papers=20):
        author_papers = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers?fields=title,authors,citationCount,abstract&limit={n_search_papers}').json()
        papers_with_annotations = [paper for paper in author_papers['data'] if paper['abstract']]
        most_important_papers = sorted(
            papers_with_annotations,
            key=lambda d: d['citationCount'], reverse=True)[:top_n_papers]
        matching_author = next(
        (author for paper in author_papers['data'] for author in paper.get('authors', []) if author.get('authorId') == str(author_id)), 
            None
            )
        return most_important_papers, matching_author
        
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

    
    def dict_to_excel(self, author_id, user_id):
        author_papers, author = self.get_author_papers(author_id)
        total_df = pd.DataFrame()

        
        for paper in author_papers:
            api_logger.info(f'{paper["paperId"]}\n')

            text_for_oecd = paper['title'] + paper['abstract']
            # Получаем данные по OECD
            oecd_answer = self.get_oecd(self.categorize_scientific_fields(text_for_oecd))
               
            # Создаём данные для DataFrame
            author_id_list = [author['authorId']] * len(oecd_answer)
            author_name = [author['name']] * len(oecd_answer)
            article_title = [paper['title']] * len(oecd_answer)
            citation_count = [paper['citationCount']] * len(oecd_answer)

            # Формируем корректный DataFrame
            answer_df = pd.DataFrame(
                data=list(zip(author_id_list, author_name, article_title, citation_count)), 
                columns=['S2 ID Автора', 'Имя Автора', 'Название статьи', 'Количество цитат']
            )

            # Объединяем с oecd_answer
            answer_df = pd.concat([answer_df, oecd_answer], axis=1)

            # Выводим результат
            total_df = pd.concat([total_df, answer_df], axis=0)

        total_df = total_df.reset_index(drop=True)
        path = f"files/{user_id}_{author_id}_oecd.xlsx"
        total_df.to_excel(path, index=False, engine='xlsxwriter')
        api_logger.info("Ответ запакован в xlsx")
        return path

    def get_answer(self, author_id, user_id):
        api_logger.info("Начало работы api")
        answer_df = self.dict_to_excel(author_id, user_id)
        api_logger.info("Файл готов к отправке")
        return answer_df
