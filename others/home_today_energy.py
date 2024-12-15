import urllib.request
import json
import time
import random
import string

# 发送 GET 请求并返回 JSON 响应的函数
def http_get(url, headers):
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"请求错误: {e}")
        return None

# 生成随机的 timestamp (Unix 时间戳，10 位)
def generate_timestamp():
    return str(int(time.time()))

# 生成随机 UUID (简单格式)
def generate_uuid():
    return '-'.join([
        ''.join(random.choices(string.hexdigits.lower(), k=8)),
        ''.join(random.choices(string.hexdigits.lower(), k=4)),
        ''.join(random.choices(string.hexdigits.lower(), k=4)),
        ''.join(random.choices(string.hexdigits.lower(), k=4)),
        ''.join(random.choices(string.hexdigits.lower(), k=12))
    ]).upper()

# 生成随机签名 (32 位十六进制字符串)
def generate_signature():
    return ''.join(random.choices(string.hexdigits.lower(), k=32))

# 主流程
def main():
    # Step 1: 获取 Token
    timestamp = generate_timestamp()
    random_value = generate_uuid()
    signature = generate_signature()
    terminalid = generate_uuid()

    login_url = f"http://ammeter0.zg118.com:9016/tenant/login/13456814595/wty20001129?timestamp={timestamp}&random={random_value}&signature={signature}"
    headers = {
        'User-Agent': 'SmartmeterTenant/1.0.2 (iPhone; iOS 18.2; Scale/3.00)',
        'Accept-Encoding': 'gzip, deflate',
        'Proxy-Connection': 'keep-alive',
        'Accept': '*/*',
        'Host': 'ammeter0.zg118.com:9016',
        'appid': '57d7b05f696960073852d2be',
        'terminalid': terminalid,
        'Connection': 'keep-alive',
        'Accept-Language': 'zh-Hans-US;q=1, en-US;q=0.9',
    }

    # 获取 Token
    token_response = http_get(login_url, headers)
    token = token_response.get("Expand", "") if token_response else ""

    # Step 2: 获取当前电表数据
    current_data_url = f"http://ammeter0.zg118.com:9016/tenant/ammeter?timestamp={generate_timestamp()}&random={generate_uuid()}&signature={generate_signature()}"
    headers['token'] = token
    headers['uid'] = '66e87a9e0ae10f03aa79e91e'

    current_data_response = http_get(current_data_url, headers)

    # Step 3: 获取历史水电数据
    today_date = time.strftime('%Y-%m-%d', time.localtime())
    history_data_url = f"http://ammeter0.zg118.com:9016/tenant/ammeter/report/60eebd0c0ae10f6077b53548/{today_date}/{today_date}?timestamp={generate_timestamp()}&random={generate_uuid()}&signature={generate_signature()}"

    history_data_response = http_get(history_data_url, headers)

    # 解析数据
    surplus, price, allpower, lasttime = 0, 0, 0, ""
    if current_data_response and 'Data' in current_data_response and len(current_data_response['Data']) > 0:
        expand = current_data_response['Data'][0].get('Expand', {})
        surplus = expand.get('surplus', 0)
        price = current_data_response['Data'][0].get('Price', 0)
        allpower = expand.get('allpower', 0)
        lasttime = expand.get('lasttime', {}).get('time', '')

    initpower, Watermoney = 0, 0
    if history_data_response and 'Data' in history_data_response and len(history_data_response['Data']) > 0:
        initpower = history_data_response['Data'][0].get('Initpower', 0)
        Watermoney = history_data_response['Data'][0].get('Watermoney', 0)

    # 计算数据
    today_electricity = allpower - initpower
    today_hotwater_L = (Watermoney / 50) * 1000

    # 结果数据
    result = {
        "token": token,
        "update_time": lasttime,
        "left_money_CNY": f"{surplus * price:.2f}",
        "today_electricity_KWh": f"{today_electricity:.2f}",
        "today_electricity_allpower_KWh": f"{allpower:.2f}",
        "today_hotwater_CNY": f"{Watermoney:.2f}",
        "today_hotwater_L": f"{today_hotwater_L:.2f}",
    }

    # 保存结果
    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    # 打印结果
    print(result)

if __name__ == "__main__":
    main()