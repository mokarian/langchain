from dotenv import load_dotenv

import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message["content"]


def call_openai():
    customer_email = """
     I'd be fuming that me blender 
    lid flew off and splattered my kitchen walls 
    with smoothie. 
    And to make matters worse, the warranty don't cover the cost of 
    cleaning up me kitchen. I need your help right now, matey.
    """

    style = "Ameriacn English \ in a calm and respoectful tone"

    prompt = f"""
    Translate the text \
    that is delimited by triple backticks into a style that is {style}.
    text: ```{customer_email}```
    """
    resp = get_completion("1 + 1")
    print(resp)


def lang_chains():
    text = """
         I'd be fuming that me blender \
        lid flew off and splattered my kitchen walls \
        with smoothie. \
        And to make matters worse, the warranty don't cover the cost of \ 
        cleaning up me kitchen. I need your help right now, matey. \
        """
    style = """
    American English \ 
    in a calm and respectful tone 
    """
    template_string = """
       Translate the text \
       that is delimited by triple backticks into a style that is: {style}.
       text: ```{text}```
       """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    # customer_messages = prompt_template.format_messages(style=style, text=text)
    chat = ChatOpenAI(temperature=0.0)
    # customer_response = chat(customer_messages)
    service_response = """
    Hey there customer, warranty \
    does not cover cleaning expenses for  
    your kitchen because it's your fault. You misused 
    your blender by forgetting to put on the lid. Tough luck. 
    See ya
    """
    service_style_pirate = """
     a polite tone that speaks in English pirate
    """
    service_messages = prompt_template.format_messages(style=service_style_pirate, text=service_response)
    # print(service_messages[0].content)
    service_response = chat(service_messages)
    print(service_response.content)
    {
        "gift": False,
        "delivery_days": 5,
        "price_value": "pretty affordable!"
    }


def json_example():
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    customer_review = """
     this 
     lead blower is pretty amazing. It has four settings, candle blower, gender 
     breeze, windy city, and tornado. 
     It arrived in two days, just in time for my wife's 
     anniversary present. 
     
     I think my wife liked it 
     so much she was speechless. So far, I've been 
     the only one using it, and I have been using it every other morning to clear the leaves on our lawn. 
     It is slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra 
     features.
     """

    review_template = """
     for the following text, extract teh following information:

     gift: Was the item purchased as a gift for someone else? 
     Answer True if it was purchased as a gift and False if not or unknown \
     delivery_days: How many days did it take for the product to be delivered? \
     price_value: Extract any sentences about the value or price \

     text:{text}
     
     {format_instructions}
     """

    schema_gifts = ResponseSchema(name="gifts", description=" Was the item purchased as a gift for someone else? "
                                                            "Answer True if it was purchased as a gift and False if not or unknown ")
    schema_delivery_days = ResponseSchema(name="delivery_days", description="How many days did it take for the product to be delivered?")
    schema_price_value = ResponseSchema(name="price_value", description="Extract any sentences about the value or price")
    response_schema = [schema_gifts, schema_delivery_days, schema_price_value]

    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions =output_parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_template(review_template)
    # print(prompt_template)
    messages = prompt_template.format_messages(text=customer_review, format_instructions = format_instructions)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    output_parser.parse(response.content)
    print(response.content)


if __name__ == '__main__':
    json_example()
