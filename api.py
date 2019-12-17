from flask import Flask, jsonify
from flask import request
from build_data import build_rule_kcs_dict, load_kcs_case_dict, build_case_pairs
from run_classifier import load_model, do_predict, flags, tf
import json

app = Flask(__name__)

customer_inputs = [
]

estimator = load_model()


def handle_rule_prioritization(customer_input, hit_rules):
    rk_dict = build_rule_kcs_dict()
    kcs_cases_df = load_kcs_case_dict()
    test_df = build_case_pairs(rk_dict, kcs_cases_df, customer_input, hit_rules)
    result_df = do_predict(estimator, test_df)
    result_df = get_filter_rank_rules(result_df)
    return result_df


def get_filter_rank_rules(result_df):
    THRESHOLD = 0.2
    kcs_rule_dict = json.load(open("./data/kcs_rule_dict.json"))
    filter_df = result_df.groupby(['caseb_kcs'])['predict_value'].agg(['mean']).reset_index()
    # filter_df[filter_df['mean'] > 0.2]
    filter_df['rule'] = filter_df.apply(lambda x: kcs_rule_dict[x['caseb_kcs']], axis=1)
    filter_df = filter_df[['rule', 'mean']]
    filter_df.columns = ['rule', 'score']
    filter_df['show'] = filter_df.apply(lambda x: True if x['score'] >= 0.2 else False, axis=1)
    filter_df = filter_df.sort_values(by=['score'], ascending=False)
    return filter_df


@app.route('/customer_inputs', methods=['GET'])
def get_customer_inputs():
    return jsonify({'customer_inputs': customer_inputs})


@app.route('/customer_inputs', methods=['POST'])
def create_customer_inputs_():
    if not request.json or not 'customer_input' in request.json or not 'hit_rules' in request.json:
        abort(400)
    input = {
        'customer_input_id': request.json['customer_input_id'],
        'customer_input': request.json['customer_input'],
        'description': request.json.get('description', ""),
        'hit_rules': request.json.get('hit_rules', []),
    }
    result_df = handle_rule_prioritization(input['customer_input'], input['hit_rules'])
    input['rank_result'] = result_df.to_json(orient='records')
    customer_inputs.append(input)

    return jsonify(input['rank_result']), 201


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.logging.set_verbosity(tf.logging.ERROR)
    app.run(host='0.0.0.0',port=8090)
