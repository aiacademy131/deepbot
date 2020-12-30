"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

import os
import sys

sys.path.append('..')

from flask import render_template

from kochat.data import Dataset
from kochat.app.scenario_manager import ScenarioManager
from kochat.loss import CRFLoss, CosFace, CenterLoss, COCOLoss, CrossEntropyLoss
from kochat.model import intent, embed, entity
from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer, SoftmaxClassifier

from demo.scenrios import restaurant, travel, dust, weather
# from scenrios import restaurant, travel, dust, weather
# 에러 나면 이걸로 실행해보세요!

from flask import Flask



# 데이터 관리
dataset = Dataset()


# 임베딩
embed_processor = GensimEmbedder(model=embed.FastText())


# 인텐트 분류
intent_classifier = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)

# 개체명 인식
entity_recognizer = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

# 시나리오
scenario_manager = ScenarioManager([
    weather, dust, travel, restaurant
])



# 학습 여부
train = False
if train:
    embed_processor.fit(dataset.load_embed())
    intent_classifier.fit(dataset.load_intent(embed_processor))
    entity_recognizer.fit(dataset.load_entity(embed_processor))



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
dialogue_cache = {}


@app.route('/')
def index():
    return render_template("index.html")





@app.route('/request_chat/<uid>/<text>', methods=['GET'])
def request_chat(uid: str, text: str) -> dict:
    
    prep = dataset.load_predict(text, embed_processor)
    intent = intent_classifier.predict(prep, calibrate=False)
    entity = entity_recognizer.predict(prep)
    text = dataset.prep.tokenize(text, train=False)
    dialogue_cache[uid] = scenario_manager.apply_scenario(intent, entity, text)

    return dialogue_cache[uid]



@app.route('/fill_slot/<uid>/<text>', methods=['GET'])
def fill_slot(uid: str, text: str) -> dict:
    prep = dataset.load_predict(text, embed_processor)
    entity = entity_recognizer.predict(prep)
    text = dataset.prep.tokenize(text, train=False)
    intent = dialogue_cache[uid]['intent']

    text = text + dialogue_cache[uid]['input']  # 이전에 저장된 dict 앞에 추가
    entity = entity + dialogue_cache[uid]['entity']  # 이전에 저장된 dict 앞에 추가

    return scenario_manager.apply_scenario(intent, entity, text)

    
    
if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
