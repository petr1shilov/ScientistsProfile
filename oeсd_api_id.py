#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import logging
import xlsxwriter
import requests
import time
from tqdm import tqdm

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


    def get_author_papers(self, author_id, top_n_papers=10, limit=100):
        api_logger.info("Начало работы для получения работ автора")
        # Получение информации об авторе
        request = requests.post(
            'https://api.semanticscholar.org/graph/v1/author/batch',
            params={'fields': 'name,hIndex,citationCount,paperCount'},
            json={"ids":[f"{author_id}"]},
        )
        author_meta = request.json()[0]
        matching_author = {'authorId': author_meta['authorId'], 'name': author_meta['name']}

        steps_num = author_meta['paperCount'] // limit + 1

        # Сбор всех статей автора
        all_papers = []
        for i in tqdm(range(steps_num), desc="Загружаю статьи автора"):
            offset = limit * i
            response = requests.get(
                f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers',
                params={
                    'fields': 'title,citationCount,authors,abstract,paperId',
                    'limit': limit,
                    'offset': offset
                },
            )
            all_papers += response.json().get('data', [])
            time.sleep(0.5)
        filtered_papers = [paper for paper in all_papers if paper.get('abstract') is not None]
        most_important_papers = sorted(filtered_papers, key=lambda x: x['citationCount'], reverse=True)[:top_n_papers]

        # Получение списка всех коавторов (для фильтрации self-citations)
        all_co_authors = set()
        for paper in most_important_papers:
            for author in paper.get('authors', []):
                if author['authorId'] and author['authorId'] != str(author_id):
                    all_co_authors.add(author['authorId'])

        # Проходим по N статьям и собираем все citing papers
        for paper_meta in most_important_papers:
            steps_num = paper_meta['citationCount'] // limit + 1

            all_citing_papers = []
            for i in tqdm(range(steps_num), desc=f"Цитируют: {paper_meta['title'][:30]}..."):
                offset = limit * i
                response = requests.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{paper_meta['paperId']}/citations",
                    params={
                        'fields': 'citingPaper.authors,citingPaper.title,citingPaper.year',
                        'limit': limit,
                        'offset': offset
                    },
                )
                all_citing_papers += response.json().get('data', [])
                time.sleep(0.5)

            # фильтрация по self-citations
            cleaned_citing_papers = []
            for citing_paper in all_citing_papers:
                authors_ids = set(authors['authorId'] for authors in citing_paper['citingPaper']['authors'] if authors['authorId'])
                if len(authors_ids & all_co_authors) == 0:
                    cleaned_citing_papers.append(citing_paper)

            paper_meta['allCitingPapers'] = len(all_citing_papers)
            paper_meta['cleanedCitingPapers'] = len(cleaned_citing_papers)
        
        api_logger.info("Обработка автора закончина")

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
        total_mectric_df = pd.DataFrame()
        
        for paper in author_papers:
            api_logger.info(f'paperId {paper["paperId"]}\n')

            title = paper['title'] or ''
            abstract = paper['abstract'] or ''
            text_for_oecd = title + abstract

            # Получаем данные по OECD
            oecd_answer = self.get_oecd(self.categorize_scientific_fields(text_for_oecd))
               
            # Создаём данные для DataFrame
            author_id_list = [author['authorId']] * len(oecd_answer)
            author_name = [author['name']] * len(oecd_answer)
            article_title = [paper['title']] * len(oecd_answer)
            citation_count = [paper['citationCount']] * len(oecd_answer)
            cleaned_citation_count = [paper['cleanedCitingPapers']] * len(oecd_answer)

            # Формируем корректный DataFrame
            answer_df = pd.DataFrame(
                data=list(zip(author_id_list, author_name, article_title, citation_count)), 
                columns=['S2 ID Автора', 'Имя Автора', 'Название статьи', 'Количество цитат']
            )

            temp_metric_df = pd.DataFrame(
                data=list(zip(article_title, citation_count, cleaned_citation_count)), 
                columns=['Название статьи', 'Количество цитат', 'Количество цитат без соавторов']
            )

            # Объединяем с oecd_answer
            answer_df = pd.concat([answer_df, oecd_answer], axis=1)
            temp_metric_df['OECD Код'] = oecd_answer['OECD Код'].values

            # Выводим результат
            total_df = pd.concat([total_df, answer_df], axis=0)
            total_mectric_df = pd.concat([total_mectric_df, temp_metric_df], axis=0)

        total_df = total_df.reset_index(drop=True)
        path = f"files/{user_id}_{author_id}_oecd.xlsx"
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            total_df.to_excel(writer, sheet_name='Статьи и области знаний', index=False)
            total_mectric_df.to_excel(writer, sheet_name='Метрики', index=False)
        api_logger.info("Ответ запакован в xlsx")
        return path

    def get_answer(self, author_id, user_id):
        api_logger.info("Начало работы api")
        answer_df = self.dict_to_excel(author_id, user_id)
        api_logger.info("Файл готов к отправке")
        return answer_df
