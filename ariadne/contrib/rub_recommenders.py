from pathlib import Path

from cassis import Cas
import more_itertools as mit
from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE, SENTENCE_TYPE

class DumbSentenceClassifier(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):

        print("\nReceived a request with the following parameters:")
        print(f"layer: {layer}")
        print(f"feature: {feature}")
        print(f"project_id: {project_id}")
        print(f"document_id: {document_id}")
        print(f"user_id: {user_id}")

        for idx, sentence in enumerate(cas.select(SENTENCE_TYPE)):
            begin = sentence.get("begin")
            end = sentence.get("end")

            if (not isinstance(begin, int) or not isinstance(end, int)):
                raise Exception("Bad parameter types")
            
            print(f"Annotating Sentence {idx}: {begin}-{end}")
            prediction = create_prediction(cas, layer, feature, begin, end, f"sent{idx}")
            cas.add_annotation(prediction)

class DumbRelClassifier(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):

        print("\nReceived a request with the following parameters:")
        print(f"layer: {layer}")
        print(f"feature: {feature}")
        print(f"project_id: {project_id}")
        print(f"document_id: {document_id}")
        print(f"user_id: {user_id}")

        sentences = cas.select(SENTENCE_TYPE)
        

        for idx, sentence in enumerate(sentences):    
            begin = sentence.get("begin")
            end = sentence.get("end")

            if (not isinstance(begin, int) or not isinstance(end, int)):
                raise Exception("Bad parameter types")      
             
            print(f"Annotating Relations in sentence {idx}")
            prediction = create_prediction(cas, layer, feature, begin, end, f"rel{idx}")
            cas.add_annotation(prediction)

class DumbChainClassifier(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):

        print("\nReceived a request with the following parameters:")
        print(f"layer: {layer}")
        print(f"feature: {feature}")
        print(f"project_id: {project_id}")
        print(f"document_id: {document_id}")
        print(f"user_id: {user_id}")
