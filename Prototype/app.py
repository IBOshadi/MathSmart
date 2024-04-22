import json
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from openai import OpenAI
from transformers import GPT2Tokenizer, GPT2LMHeadModel

client = OpenAI(api_key="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")
job_id="ftjob-l76BypI3HNeM7qa7zewmwavt"
response = client.fine_tuning.jobs.retrieve(job_id)#job id

def format_test(question):
    formatted_message = [{"role": "user", "content": question}]
    return formatted_message

def predict(question, fine_tuned_model_id):
    test_message = format_test(question)
    response = client.chat.completions.create(
        model=fine_tuned_model_id, messages=test_message, temperature=1, max_tokens=250
    )
    return response.choices[0].message.content

def store_predictions(question, fine_tuned_model_id):
    prediction_result = predict(question, fine_tuned_model_id)
    return prediction_result

temperature=1
max_tokens=500

def response_gpt3(model,question):
  response = client.completions.create(
      model=model,
      prompt=question,
      max_tokens=max_tokens,
      temperature=temperature
      )

def response_gpt2(model_name,input_question): 
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(f"[Q] {input_question}", return_tensors="pt")

    # Generate the equation
    output = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0)
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    index_a = generated_answer.find("[A]")
    answer = generated_answer[index_a + 4:]

    return answer

app = Flask('app')
CORS(app, resource={r"/api/*": {"origins": "*"}})
app.config['CORS HEADERS'] = 'content-Type'


@app.route('/api/getPrediction', methods=['POST'])
@cross_origin()
def get_prediction():
  if request.method == 'POST':
    try:
      data = request.json
      user_input = data.get('userInput')
      selectedModel = data.get('selectedModel')

      # pre process implementation
      if selectedModel == 'gpt2':
        model_name = 'gpt2'
        model_name_new = '/D1000'
        # edit accordingly
        prediction = response_gpt2(model_name,user_input)
        fine_tuned_prediction = response_gpt2(model_name_new,user_input)
      elif selectedModel == 'gpt3':
        model = "davinci-002"
        FINE_TUNED_MODEL = response.fine_tuned_model
        # edit accordingly

        prediction = response_gpt3(model,user_input)
        fine_tuned_prediction = response_gpt3(FINE_TUNED_MODEL,user_input)
      else:
        # edit accordingly
        model="gpt-3.5-turbo"
        fine_tuned_model_id = response.fine_tuned_model
        prediction = store_predictions(user_input, model)
        fine_tuned_prediction = store_predictions(user_input, fine_tuned_model_id)
      

      return json.dumps({
        "responseText1": prediction,
        "responseText2": fine_tuned_prediction
      })
    except Exception as e:
      error_message = f"Internal Server Error: {str(e)}"
      print(error_message)
      return json.dumps({
        "status": 500,
        "message": error_message
      })

if __name__ == '__main__':
    app.run()