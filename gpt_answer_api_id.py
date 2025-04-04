#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import logging
import requests
import xlsxwriter
from tqdm import tqdm

from bot.texts import *

import config

api_logger = logging.getLogger("api_logger")

# Настраиваем формат и уровень логирования только для нашего логгера
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s - %(message)s")
handler.setFormatter(formatter)
api_logger.addHandler(handler)
api_logger.setLevel(logging.INFO)


class GPTGenApiId:
    def __init__(self, api_key=config.api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model

        self.client = OpenAI(api_key=self.api_key)

    def excel_pre_processing(self, file_path) -> list:
        api_logger.info('Работа с excel')
        df = pd.read_excel(file_path)
        if df.empty:
            return 'DataFrame is empty'
        else:
            annotation_columns = [col for col in df.columns if 'аннотация' in col.lower()]
            annotation_dict = df.set_index('Название лаборатории / центра')[annotation_columns].apply(lambda row: row.tolist(), axis=1).to_dict()
            return annotation_dict   

    def get_author_meta(self, author_id, max_retries=5, wait_seconds=3):
        url = 'https://api.semanticscholar.org/graph/v1/author/batch'
        params = {'fields': 'name,hIndex,citationCount,paperCount'}
        payload = {"ids": [author_id]}

        for attempt in range(1, max_retries + 1):
            api_logger.info(f"Попытка {attempt} запроса для автора {author_id}...")

            try:
                response = requests.post(url, params=params, json=payload)
                if response.status_code != 200:
                    api_logger.info(f"HTTP {response.status_code}: {response.text}")
                    time.sleep(wait_seconds)
                    continue

                data = response.json()

                # Проверка: API вернул список?
                if isinstance(data, list) and data:
                    api_logger.info("Успешно получен ответ")
                    return data[0]

                # Иногда API может вернуть {'error': ...}
                elif isinstance(data, dict) and "error" in data:
                    api_logger.info(f"API error: {data['error']}")
                else:
                    api_logger.info("⚠️ Ответ API не является списком или пуст")

            except Exception as e:
                api_logger.info(f"Ошибка запроса: {e}")

            time.sleep(wait_seconds)  # подождать перед новой попыткой

        api_logger.info("⛔ Все попытки исчерпаны. Ответ не получен.")
        return None     
        
    def get_author_papers(self, author_id, top_n_papers=10, limit=100):
        api_logger.info("Начало работы для получения работ автора")
        # Получение информации об авторе
        author_meta = self.get_author_meta(author_id)
        if author_meta:
            matching_author = {'authorId': author_meta['authorId'], 'name': author_meta['name']}
        else: 
            api_logger.info('Данные не загрузились')

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

        list_of_abstracts = [i['abstract'] for i in most_important_papers]
        annotation_dict = {matching_author['name']: list_of_abstracts}
        print(annotation_dict)
        return annotation_dict
      

    def candidat_search(self, annotation_dict):
        start_time = time.time()
        prompt_token_sum = 0
        completion_token_sum = 0
        candidates_dict = {}

        for lab_name, annotations in annotation_dict.items():
            api_logger.info(f'Обработка аннотаций из лаборатории {lab_name}')
            lab_candidates = []

            for annotation_id, annotation in enumerate(annotations):
                api_logger.info(f'Обработка аннотации {annotation_id} из {len(annotations) - 1} из {lab_name}')
                messages = [
                    {"role": "system", "content": candidates_system_prompt},
                    {"role": "user", "content": annotation}
                ]
                
                response = self.client.chat.completions.create(
                            model=self.model, messages=messages)
                
                answer = response.choices[0].message.content
                prompt_token_sum += response.usage.prompt_tokens
                completion_token_sum += response.usage.completion_tokens
                lab_candidates.append(answer)  

            candidates_dict[lab_name] = lab_candidates  
        api_logger.info(
            f'Обработаны все лаборатории за {time.time() - start_time:.2f} секунд\n'
            f'Токенов потрачено:\n\tна промпт -> {prompt_token_sum}\n\tна ответ -> {completion_token_sum}'
        )
        return candidates_dict, prompt_token_sum, completion_token_sum

    def total_answer(self, candidates_dict, prompt_token_sum, completion_token_sum):
        start_time = time.time()
        total_answer = {}
        for lab_name, annotations in candidates_dict.items():
            api_logger.info(f'Обработка кандидатов из лаборатории {lab_name}')
            messages = [
                    {"role": "system", "content": total_system_prompt},
                    {"role": "user", "content": '\n'.join(annotations)}
                ]
                
            response = self.client.chat.completions.create(
                            model=self.model, messages=messages)
            answer = response.choices[0].message.content
            prompt_token_sum += response.usage.prompt_tokens
            completion_token_sum += response.usage.completion_tokens

            total_answer[lab_name] = answer
        
        api_logger.info(
            f'Обработаны все лаборатории за {time.time() - start_time:.2f} секунд\n'
            f'Токенов потрачено:\n\tна промпт -> {prompt_token_sum}\n\tна ответ -> {completion_token_sum}'
        )
        
        return total_answer



    def dict_to_excel(self, doc, author_id, user_id):
        name_labs, total_candidats = zip(*doc.items())

        list_of_total_candidats = ['\n'.join(candidat.replace('\n"','').replace('"','').split(',')).replace('[','').replace(']','') for candidat in total_candidats]

        df = pd.DataFrame({
            'Имя Автора': name_labs,
            'Перечень компетенций / областей научных интересов Автора': list_of_total_candidats
        })

        path = f"files/{user_id}_{author_id}_gpt.xlsx"
        df.to_excel(path, index=False, engine='xlsxwriter')
        api_logger.info("Ответ запакован в xlsx")
        return path

    def get_answer(self, author_id, user_id):
        api_logger.info("Начало работы api")
        annotation_dict = self.get_author_papers(author_id)
        candidates_dict, prompt_token_sum, completion_token_sum = self.candidat_search(annotation_dict)
        answer = self.total_answer(candidates_dict, prompt_token_sum, completion_token_sum)
        answer_df = self.dict_to_excel(answer, author_id, user_id)
        api_logger.info("Файл готов к отправке")
        return answer_df
