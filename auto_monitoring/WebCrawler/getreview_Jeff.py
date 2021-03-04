import requests
import csv
# 字典型,代理
headers_meituan = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"
}


def get_ratings(id):
    # 创建文件夹并打开
    fp = open("./美团评论_{}.csv".format(id), 'a',
              newline='', encoding='utf-8-sig')
    writer = csv.writer(fp)  # 我要写入
    # 写入内容
    writer.writerow(("用户", "ID", "链接", "评论"))  # 运行一次

    for num in range(0, 381, 10):
        print("正在爬取%s条............" % num)
        ajax_url = "https://www.meituan.com/meishi/api/poi/getMerchantComment?uuid=60d8293a-8f06-4d5e-b13d-796bbda5268f&platform=1&partner=126&originUrl=https%3A%2F%2Fwww.meituan.com%2Fmeishi%2F{}%2F&riskLevel=1&optimusCode=10&id={}&userId=&offset=" + \
            str(num) + "&pageSize=10&sortType=1"
        ajax_url = ajax_url.format(id, id)
        print(ajax_url)
        reponse = requests.get(url=ajax_url, headers=headers_meituan)
        # 按ctrl+},往右边回退
        print(reponse)
        for item in reponse.json()["data"]["comments"]:
            name = item["userName"]
            user_id = item["userId"]
            user_url = item["userUrl"]
            comment = item["comment"]
            result = (name, user_id, user_url, comment)
            writer.writerow(result)
    fp.close()


if __name__ == '__main__':
    get_ratings(157841870)
