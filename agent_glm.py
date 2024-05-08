import os
from zhipuai import ZhipuAI
from IPython.display import display, Code, Markdown
import json
import tiktoken
import time


# 外部函数python解释器
def python_inter(py_code, g='globals()'):
    """
    专门用于执行非绘图类python代码，并获取最终查询或处理结果。若是设计绘图操作的Python代码，则需要调用fig_inter函数来执行。
    :param py_code: 字符串形式的Python代码，用于执行对telco_db数据库中各张数据表进行操作
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果
    """

    global_vars_before = set(g.keys())
    try:
        exec(py_code, g)
    except Exception as e:
        return f"代码执行时报错{e}"
    global_vars_after = set(g.keys())
    new_vars = global_vars_after - global_vars_before
    # 若存在新变量
    if new_vars:
        result = {var: g[var] for var in new_vars}
        return str(result)
    # 若不存在新变量，即有可能是代码是表达式，也有可能代码对相同变量重复赋值
    else:
        try:
            # 尝试如果是表达式，则返回表达式运行结果
            return str(eval(py_code, g))
        # 若报错，则先测试是否是对相同变量重复赋值
        except Exception as e:
            try:
                exec(py_code, g)
                return "已经顺利执行代码"
            except Exception as e:
                pass
            # 若不是重复赋值，则报错
            return f"代码执行时报错{e}"


def get_glm_response(messages,
                     tools=None,
                     model='glm-4'):
    """
    单次GLM模型响应函数，能够正常获取模型响应，并在模型无法正常响应时暂停模型调用，\
    并在休息一分钟之后继续调用模型。最多尝试三次。
    :param messages: 必要参数，字典类型，输入到Chat模型的messages参数对象
    :param tools: 可选参数，默认为None，可以设置为包含全部外部函数参数解释Schema格式列表
    :param model: Chat模型，可选参数，默认模型为glm-4
    :return：Chat模型输出结果
    """
    attempts = 0
    max_attempts = 3

    api_key = os.getenv("ZHIPU_API_KEY")
    client = ZhipuAI(api_key=api_key)

    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # 使用函数参数
                tools=tools,
            )
            return response  # 成功时返回响应
        except Exception as e:  # 捕获所有异常
            print(f"发生错误：{e}")
            attempts += 1
            if attempts < max_attempts:
                print("等待1分钟后重试...")
                time.sleep(60)  # 等待1分钟
            else:
                print("尝试次数过多，停止尝试。")
                return None  # 在所有尝试后返回 None


def check_code_run(messages,
                   functions_list=None,
                   tools=None,
                   model="glm-4",
                   auto_run=True):
    """
    能够自动执行外部函数调用的Chat对话模型，专门用于代码解释器的构建过程，可以通过auto_run参数设置，决定是否自动执行代码
    :param messages: 必要参数，字典类型，输入到Chat模型的messages参数对象
    :param functions_list: 可选参数，默认为None，可以设置为包含全部外部函数的列表对象
    :param tools: 可选参数，默认为None，可以设置为包含全部外部函数参数解释Schema格式列表
    :param model: Chat模型，可选参数，默认模型为glm-4
    :auto_run：在调用外部函数的情况下，是否自动进行Second Response。该参数只在外部函数存在时起作用
    :return：Chat模型输出结果
    """

    # 如果没有外部函数库，则执行普通的对话任务
    if tools is None:
        response = get_glm_response(
            messages=messages,
            tools=tools,
            model=model
        )
        response_message = response.choices[0].message

    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # first response
        response = get_glm_response(
            messages=messages,
            tools=tools,
            model=model
        )
        response_message = response.choices[0].message

        # 判断返回结果是否存在function_call，即判断是否需要调用外部函数来回答问题
        # 若存在function_call，则执行Function calling流程
        # 需要调用外部函数，由于考虑到可能存在多次Function calling情况，这里创建While循环
        # While循环停止条件：response_message不包含function_call
        while response_message.tool_calls is not None:
            print("正在调用外部函数...")
            # 获取函数名
            function_name = response_message.tool_calls[0].function.name
            # 获取函数对象
            fuction_to_call = available_functions[function_name]
            try:
                # 获取函数参数
                function_args = json.loads(response_message.tool_calls[0].function.arguments)
                # 将当前操作空间中的全局变量添加到外部函数中
                function_args['g'] = globals()

                def convert_to_markdown(code, language):
                    return f"```{language}\n{code}\n```"

                if function_args.get('py_code'):
                    code = function_args['py_code']
                    markdown_code = convert_to_markdown(code, 'python')
                    print("即将执行以下代码：")

                else:
                    markdown_code = function_args

                display(Markdown(markdown_code))

                if not auto_run:
                    res = input('请确认是否运行上述代码（1），或者退出本次运行过程（2）')

                    if res == '2':
                        print("终止运行")
                        return None

                print("正在执行代码，请稍后...")

                # 将函数参数输入到函数中，获取函数计算结果
                function_response = fuction_to_call(**function_args)

                print("外部函数已运行完毕")

                # messages中拼接first response消息
                messages.append(response_message.model_dump())
                # messages中拼接函数输出结果
                messages.append(
                    {
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": response_message.tool_calls[0].id
                    }
                )

                # 第二次调用模型
                second_response = get_glm_response(
                    messages=messages,
                    tools=tools,
                    model=model
                )

                # 更新response_message
                response_message = second_response.choices[0].message

            except Exception as e:
                print("json格式对象创建失败，正在重新运行")
                messages = check_code_run(messages=messages,
                                          functions_list=functions_list,
                                          tools=tools,
                                          model=model,
                                          auto_run=auto_run)
                return messages

    # While条件不满足，或执行完While循环之后，提取返回结果
    final_response = response_message

    display(Markdown(final_response.content))

    messages.append(final_response.model_dump())

    return messages


# 编写python解释器方法说明
python_inter_function_info = {
    'name': 'python_inter',
    'description': '专门用于python代码，并获取最终查询或处理结果。',
    'parameters': {
        'type': 'object',
        'properties': {
            'py_code': {
                'type': 'string',
                'description': '用于执行在本地环境运行的python代码'
            },
            'g': {
                'type': 'string',
                'description': '环境变量，可选参数，保持默认参数即可'
            }
        },
        'required': ['py_code']
    }
}

# GLM-4模型的Function calling封装为一个tools对象，同时说明当前Function性质
tools = [
    {
        "type": "function",
        "function": python_inter_function_info
    }
]

# 打开并读取Markdown文件
with open('telco_data_dictionary.md', 'r', encoding='utf-8') as f:
    data_dictionary = f.read()

messages = [
    {"role": "system", "content": data_dictionary},
    {"role": "user", "content": "请帮我读取telco_data数据集到本地python环境，并将其命名为data"},
]


# 多轮对话函数编写
def chat_with_inter(prompt="你好呀",
                    model="glm-4",
                    functions_list=None,
                    tools=None,
                    system_content='',
                    auto_run=True,
                    history_record=False):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # 多轮对话阈值
    tokens_thr = 120000

    messages = []

    if system_content != '':
        messages = [{"role": "system", "content": system_content}]

    messages.append({"role": "user", "content": prompt})

    tokens_count = len(encoding.encode((prompt + system_content)))

    while True:

        messages = check_code_run(messages=messages,
                                  functions_list=functions_list,
                                  tools=tools,
                                  model=model,
                                  auto_run=auto_run)

        # 询问用户是否还有其他问题
        user_input = input("您还有其他问题吗？(输入退出以结束对话): ")
        if user_input == "退出":
            del messages
            break

        answer = messages[-1]["content"]

        # 记录新一轮问答
        messages.append({"role": "user", "content": user_input})

        # 计算当前总token数
        tokens_count += len(encoding.encode((answer + user_input)))

        # 删除超出token阈值的对话内容，删除第二条消息
        while tokens_count >= tokens_thr:
            tokens_count -= len(encoding.encode(messages.pop(1)["content"]))

    if history_record:
        return messages


chat_with_inter(prompt="请帮我介绍下telco_data数据集",
                        functions_list=[python_inter],
                        tools = tools,
                        system_content=data_dictionary)
