import requests
import json
import time
import uuid
import random
import string
import pandas as pd

# 发送 GET 请求并返回 JSON 响应的函数
def http_get(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # 如果响应状态码不是 200，抛出异常
        return response.json()  # 返回 JSON 响应
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

# 生成随机的 timestamp (Unix 时间戳，10 位)
def generate_timestamp():
    return str(int(time.time()))

# 生成随机 UUID (标准 UUID 格式，用于 random 和 terminalid)
def generate_uuid():
    return str(uuid.uuid4()).upper()

# 生成随机的签名 (32 位十六进制字符串)
def generate_signature():
    return ''.join(random.choices(string.hexdigits.lower(), k=32))

# 主流程
def main():
    # Step 1: 获取 Token
    timestamp = generate_timestamp()  # 随机生成 timestamp
    random_value = generate_uuid()  # 随机生成 random 值
    signature = generate_signature()  # 随机生成签名
    terminalid = generate_uuid()  # 随机生成 terminalid

    login_url = f"http://ammeter0.zg118.com:9016/tenant/login/13456814595/wty20001129?timestamp={timestamp}&random={random_value}&signature={signature}"
    
    headers = {
        'User-Agent': 'SmartmeterTenant/1.0.2 (iPhone; iOS 18.1; Scale/3.00)',
        'Accept-Encoding': 'gzip, deflate',
        'Proxy-Connection': 'keep-alive',
        'Accept': '*/*',
        'Host': 'ammeter0.zg118.com:9016',
        'appid': '57d7b05f696960073852d2be',
        'terminalid': terminalid,
        'Connection': 'keep-alive',
        'Accept-Language': 'zh-Hans-US;q=1, en-US;q=0.9',
    }

    # 发送获取 Token 的请求
    token_response = http_get(login_url, headers)
    if token_response:
        token = token_response.get("Expand", "")
        print(f"Token: {token}")
    else:
        print("获取 Token 失败")
        return

    # Step 2: 使用 Token 获取当前电表数据
    current_data_url = f"http://ammeter0.zg118.com:9016/tenant/ammeter?timestamp={generate_timestamp()}&random={generate_uuid()}&signature={generate_signature()}"
    
    headers['token'] = token  # 将 token 添加到请求头
    headers['uid'] = '66e87a9e0ae10f03aa79e91e'  # 使用样例的 UID

    # 发送获取当前电表数据的请求
    current_data_response = http_get(current_data_url, headers)
    if current_data_response:
        print("当前电表数据:")
        print(json.dumps(current_data_response, indent=2))
    else:
        print("获取当前电表数据失败")
        return

    # Step 3: 使用 Token 获取历史水电数据
    history_data_url = f"http://ammeter0.zg118.com:9016/tenant/ammeter/report/60eebd0c0ae10f6077b53548/2024-03-18/2024-12-31?timestamp={generate_timestamp()}&random={generate_uuid()}&signature={generate_signature()}"

    # 发送获取历史数据的请求
    history_data_response = http_get(history_data_url, headers)
    if history_data_response:
        print("历史水电数据:")
        print(json.dumps(history_data_response, indent=2))
    else:
        print("获取历史水电数据失败")
        return

    # Step 4: 将所有响应数据合并成一个 JSON 对象
    combined_data = {
        "token": token,
        "currentData": current_data_response,
        "historyData": history_data_response
    }

    # 将合并的数据写入文件
    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(combined_data, indent=2, ensure_ascii=False))

    print(f"数据已写入 {output_file}")

    # Step 5: 将 JSON 数据转换为 CSV 并保存
    convert_json_to_csv(combined_data)

# JSON 转 CSV 的函数
def convert_json_to_csv(json_data):
    # 提取 `currentData` 中的 "Data" 部分
    current_data_list = json_data['currentData']['Data']
    history_data_list = json_data['historyData']['Data']

    # 将 JSON 数据转换为 pandas DataFrame
    current_df = pd.DataFrame(current_data_list)
    history_df = pd.DataFrame(history_data_list)

    # 保存为 CSV 文件
    current_csv_file = "current_data.csv"
    history_csv_file = "history_data.csv"

    current_df.to_csv(current_csv_file, index=False, encoding='utf-8')
    history_df.to_csv(history_csv_file, index=False, encoding='utf-8')

    print(f"当前数据已保存为 {current_csv_file}")
    print(f"历史数据已保存为 {history_csv_file}")

# 调用主流程
if __name__ == "__main__":
    main()