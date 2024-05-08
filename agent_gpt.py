import json
import os
import requests
from openai import AzureOpenAI


def run_conv(messages,
             functions_list=None,
             functions=None,
             model="gpt-4-0125",):
    """
    能够自动执行外部函数调用的Chat对话模型
    :param messages: 必要参数，输入到Chat模型的messages参数对象
    :param functions_list: 可选参数，默认为None，可以设置为包含全部外部函数的列表对象
    :param model: Chat模型，可选参数，默认模型为gpt-3.5-turbo-0613
    :return：Chat模型输出结果
    """

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    user_messages = messages

    # 如果没有外部函数库，则执行普通的对话任务
    if functions_list is None:
        response = client.chat.completions.create(
            model=model,
            timeout=1000,
            messages=user_messages
        )
        response_message = response.choices[0].message.content
        final_response = response_message["content"]

    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # 创建包含用户问题的message
        messages = user_messages

        # first response
        response = client.chat.completions.create(
            model=model,
            timeout=1000,
            messages=messages,
            functions=functions,
            function_call="auto"
        )

        response_message = response.choices[0].message.function_call

        # 获取函数名
        function_name = response_message.name
        # 获取函数对象
        fuction_to_call = available_functions[function_name]
        # 获取函数参数
        function_args = json.loads(response_message.arguments)

        # 将函数参数输入到函数中，获取函数计算结果
        function_response = fuction_to_call(**function_args)

        # messages中拼接函数输出结果
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        # 第二次调用模型
        second_response = client.chat.completions.create(
            model=model,
            timeout=1000,
            messages=messages
        )
        # 获取最终结果
        final_response = second_response.choices[0].message.content

    return final_response


def get_weather(location):
    """
    查询即时天气函数
    :param location: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": location,
        "appid": "9fe3eee2165574879be2fbf8423499dd",  # 输入API key
        "units": "metric",  # 使用摄氏度而不是华氏度
        "lang": "zh_cn"  # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)

    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)


get_weather_function = {
    "type": "function",
    'name': 'get_weather',
    'description': '查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'description': "城市名称，注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则location参数需要输入'Beijing'",
                'type': 'string'
            }
        },
        'required': ['location']
    }
}

# 同时还需要封装外部函数库，用于关联外部函数名称和外部函数对象
available_functions = {
    "get_weather": get_weather,
}

messages = [{"role": "user", "content": "请问郑州今天天气如何？"}]
result = run_conv(messages=messages, functions_list=[get_weather], functions=[get_weather_function])

# result = run_conv([{"role": "user", "content": "请问什么是机器学习？"}])
print(result)
