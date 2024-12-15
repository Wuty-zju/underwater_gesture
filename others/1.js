const http = require("http");
const fs = require("fs");

// 发送 GET 请求并返回 JSON 响应的函数
function httpGet(url, headers) {
  return new Promise((resolve, reject) => {
    const options = {
      headers: headers,
    };

    http
      .get(url, options, (res) => {
        let data = "";

        res.on("data", (chunk) => {
          data += chunk;
        });

        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(`解析 JSON 错误: ${e.message}`);
          }
        });
      })
      .on("error", (err) => {
        reject(`请求错误: ${err.message}`);
      });
  });
}

// 生成随机的 timestamp (Unix 时间戳，10 位)
function generateTimestamp() {
  return Math.floor(Date.now() / 1000).toString();
}

// 生成随机 UUID (标准 UUID 格式，用于 random 和 terminalid)
function generateUUID() {
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11)
    .replace(/[018]/g, (c) =>
      (c ^ (Math.random() * 16) >> (c / 4)).toString(16)
    )
    .toUpperCase();
}

// 生成随机的签名 (32 位十六进制字符串)
function generateSignature() {
  const characters = "0123456789abcdef";
  let signature = "";
  for (let i = 0; i < 32; i++) {
    signature += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return signature;
}

// 主流程
async function main() {
  // Step 1: 获取 Token
  const timestamp = generateTimestamp(); // 随机生成 timestamp
  const randomValue = generateUUID(); // 随机生成 random 值
  const signature = generateSignature(); // 随机生成签名
  const terminalId = generateUUID(); // 随机生成 terminalid

  const loginUrl = `http://ammeter0.zg118.com:9016/tenant/login/13456814595/wty20001129?timestamp=${timestamp}&random=${randomValue}&signature=${signature}`;

  const headers = {
    "User-Agent": "SmartmeterTenant/1.0.2 (iPhone; iOS 18.2; Scale/3.00)",
    "Accept-Encoding": "gzip, deflate",
    "Proxy-Connection": "keep-alive",
    Accept: "*/*",
    Host: "ammeter0.zg118.com:9016",
    appid: "57d7b05f696960073852d2be",
    terminalid: terminalId,
    Connection: "keep-alive",
    "Accept-Language": "zh-Hans-US;q=1, en-US;q=0.9",
  };

  try {
    const tokenResponse = await httpGet(loginUrl, headers);
    const token = tokenResponse && tokenResponse.Expand ? tokenResponse.Expand : "";

    // Step 2: 使用 Token 获取当前电表数据
    const currentDataUrl = `http://ammeter0.zg118.com:9016/tenant/ammeter?timestamp=${generateTimestamp()}&random=${generateUUID()}&signature=${generateSignature()}`;
    headers["token"] = token;
    headers["uid"] = "66e87a9e0ae10f03aa79e91e"; // 使用样例的 UID

    const currentDataResponse = await httpGet(currentDataUrl, headers);

    // Step 3: 使用 Token 获取历史水电数据
    const todayDate = new Date().toISOString().split("T")[0];
    const historyDataUrl = `http://ammeter0.zg118.com:9016/tenant/ammeter/report/60eebd0c0ae10f6077b53548/${todayDate}/${todayDate}?timestamp=${generateTimestamp()}&random=${generateUUID()}&signature=${generateSignature()}`;

    const historyDataResponse = await httpGet(historyDataUrl, headers);

    // Step 4: 将所有响应数据合并成一个 JSON 对象
    const combinedData = {
      token,
      currentData: currentDataResponse,
      historyData: historyDataResponse,
    };

    // 将合并的数据写入文件
    const outputFile = "output.json";
    fs.writeFileSync(outputFile, JSON.stringify(combinedData, null, 2), "utf-8");

    // 解析 combinedData 中的 surplus 和 Price，Allpower，Initpower
    let surplus = 0,
      price = 0,
      allpower = 0,
      lasttime = "";

    if (currentDataResponse && currentDataResponse.Data && currentDataResponse.Data.length > 0) {
      const expand = currentDataResponse.Data[0].Expand || {};
      surplus = expand.surplus || 0;
      price = currentDataResponse.Data[0].Price || 0;
      allpower = expand.allpower || 0;
      lasttime = expand.lasttime && expand.lasttime.time ? expand.lasttime.time : "";
    }

    let initpower = 0,
      Watermoney = 0;
    if (historyDataResponse && historyDataResponse.Data && historyDataResponse.Data.length > 0) {
      initpower = historyDataResponse.Data[0].Initpower || 0;
      Watermoney = historyDataResponse.Data[0].Watermoney || 0;
    }

    // 计算 today
    const todayElectricity = allpower - initpower;
    const todayElectricityAllpower = allpower;
    const todayHotwaterL = (Watermoney / 50) * 1000;

    // 保存为字典
    const today = {
      token,
      update_time: lasttime,
      left_money_CNY: (surplus * price).toFixed(2),
      today_electricity_KWh: todayElectricity.toFixed(2),
      today_electricity_allpower_KWh: todayElectricityAllpower.toFixed(2),
      today_hotwater_CNY: Watermoney.toFixed(2),
      today_hotwater_L: todayHotwaterL.toFixed(2),
    };

    // 打印结果
    console.log(today);
  } catch (error) {
    console.error(`发生错误: ${error}`);
  }
}

// 调用主流程
main();
