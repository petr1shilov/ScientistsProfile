#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import logging
import requests
import xlsxwriter

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
        
    def get_author_papers(self, author_id, top_n_papers=2, n_search_papers=20):
        author_papers = author_papers = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers?fields=title,authors,citationCount,abstract&limit={n_search_papers}').json()
        papers_with_annotations = [paper for paper in author_papers['data'] if paper['abstract']]
        most_important_papers = sorted(
            papers_with_annotations,
            key=lambda d: d['citationCount'], reverse=True)[:top_n_papers]
        matching_author = next(
        (author for paper in author_papers['data'] 
                for author in paper.get('authors', []) 
                    if author.get('authorId') == str(author_id)), 
                None
            )
        list_of_abstracts = [i['abstract'] for i in most_important_papers]
        annotation_dict = {matching_author['name']: list_of_abstracts}
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
