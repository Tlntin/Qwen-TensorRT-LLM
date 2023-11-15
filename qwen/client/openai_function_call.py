from openai import OpenAI
import requests
import urllib3
import time
import random
import json


urllib3.disable_warnings()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no api"
)

# get api from here https://dev.qweather.com/
weather_key = ""
assert len(weather_key) > 0, print("please get weather query api in https://dev.qweather.com/")


class Weather:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_location_from_api(self, location, adm=None,
                              location_range="world", lang="zh"):
        """
        Get api based on Japanese weather
        params location: the location to be queried
        params adm: superior region, for example, the superior region of Yuexiu is Guangzhou
        params location_range: query range, default global, supports cn: China, us: United States, fr: France,
        uk: United Kingdom, please check the iso-3166 standard for more information
        params lang: language, default zh, support en
        """
        url = "https://geoapi.qweather.com/v2/city/lookup?"
        params = {
            "key": self.api_key,
            "location": location,
            "range": location_range,
            "lang": lang,
        }
        if adm is not None:
            if len(adm) > 0:
                params["adm"] = adm
        session = requests.session()
        try:
            res2 = session.get(url, params=params, verify=False, timeout=15)
            if res2.status_code == 200:
                data = res2.json()
                if data.get("code", None) == '200':
                    return data.get("location", [])
                else:
                    print(data)
            else:
                print(res2)
            time.sleep(1 + random.random())
            session.close()
        except Exception as err:
            print("request error", err)
            time.sleep(3 + random.random())
            session.close()
        return []

    def get_weather_from_api(self, location: str):
        """
        Get weather information from Zefeng weather api
        :param location: location information, which can be location_id or a latitude and longitude (format: "longitude, latitude")
        """
        url = "https://devapi.qweather.com/v7/weather/3d?"
        params = {
            "location": location,
            "key": self.api_key
        }
        session = requests.session()
        try:
            res1 = session.get(url, params=params, verify=False, timeout=15)
            if res1.status_code == 200:
                data = res1.json()
                if data.get("code", "") == "200":
                    return data.get("daily", [])
                else:
                    print(data)
            else:
                print(res1)
            time.sleep(1 + random.random())
            session.close()
        except Exception as err:
            print("get api error，", err)
            time.sleep(3 + random.random())
            session.close()
        return []


def get_current_weather(location: str):
    weather = Weather(weather_key)
    location_data = weather.get_location_from_api(location)
    if len(location_data) > 0:
        location_dict = location_data[0]
        city_id = location_dict["id"]
        weather_res = weather.get_weather_from_api(city_id)
        n_day = len(weather_res)
        return f"查询到最近{n_day}天的天气。" + json.dumps(weather_res, ensure_ascii=False)
    else:
        return ""

def call_qwen(messages, functions=None):
    # print(messages)
    if functions:
        response = client.chat.completions.create(
            model="Qwen", messages=messages, functions=functions
        )
    else:
        response = client.chat.completions.create(
            model="Qwen", messages=messages
        )
    # print(response)
    # print(response.choices[0].message.content)
    return response


def chat(query: str):
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            # Note: The current version of Qwen-7B-Chat (as of 2023.08) performs okay with Chinese tool-use prompts,
            # but performs terribly when it comes to English tool-use prompts, due to a mistake in data collecting.
            "content": query,
        }
    ]
    response = call_qwen(messages, functions)
    res = response.choices[0].message
    message_dict = {
        "role": res.role,
        "content": res.content,
        "function_call": res.function_call,
    }
    messages.append(message_dict)
    # --- call function --- #
    if res.function_call is not None:
        function_call = res.function_call
        function_name = function_call.name
        try:
            function_params = json.loads(function_call.arguments)
        except:
            print(f"{function_name}解析对应参数失败，请检查, 参数信息：", function_call)
            return
        for temp_dict in functions:
            if temp_dict["name"] == function_name:
                require_params = temp_dict["parameters"]["required"]
                require_params.sort()
                had_params = list(function_params.keys())
                had_params.sort()
                if had_params != require_params:
                    # need to do other fill params
                    return

                response = eval(function_name)(**function_params)
                message = {
                    "role": "function",
                    "name": function_name,
                }
                if len(response) > 0:
                    message["content"] = response
                else:
                    message["content"] = "未找到任何信息"
                messages.append(message)
                response = call_qwen(messages, functions)
    return response


messages = [{"role": "system", "content": "You are a helpful assistant."}]
print("=" * 20)
print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
print("目前已支持天气查询插件")
print("=" * 20)
query = "北京天气如何？穿短袖会不会冷？"
print("用户输入：", query)
res = chat(query)
print("回答结果：", res.choices[0].message.content)