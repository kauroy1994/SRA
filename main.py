# Python program to read
# json file
import logging
import pandas as pd
from simpletransformers.seq2seq import(Seq2SeqModel,Seq2SeqArgs,)

import json

def read_json():
    # Opening JSON file
    f = open('human_annotations.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    return data

def main():
    # Iterating through the json
    # list
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    data = read_json()

    train_data = []
    for i in range(1):
        pair = [data['data'][i]['text'],data['data'][i+1]['text']]
        train_data.append(pair)

    train_df = pd.DataFrame(
        train_data, columns=["input_text", "target_text"]
    )

    eval_data = train_data

    eval_df = pd.DataFrame(
        eval_data, columns=["input_text", "target_text"]
    )

    model_args = Seq2SeqArgs()
    model_args.num_train_epochs = 10
    model_args.no_save = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        use_cuda=False,
    )
    model.train_model(
    train_df, eval_data=eval_df, matches=count_matches
    )
    results = model.eval_model(eval_df)

    print(
        model.predict(
            [
                "You sound like an animal lover too.  Any pets?"
            ]
        )
    )

def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )

if __name__ == '__main__':
    main()



