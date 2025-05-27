import requests
import json


def get_taxon_info(ott_id):


    # API 端点 URL
    url = "https://api.opentreeoflife.org/v3/taxonomy/taxon_info"

    # 请求头，指定内容类型为 JSON
    headers = {
        "Content-Type": "application/json"
    }

    # 请求体，包含要查询的 ott_id
    data = {
        "ott_id": ott_id
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data)

        # 检查响应状态码
        if response.status_code == 200:
            # 请求成功，解析 JSON 响应
            result = response.json()

            # 打印结果（这里只展示部分关键信息，您可以根据需要调整）
            print(f"分类单元名称: {result.get('name')}")
            print(f"OTT ID: {result.get('ott_id')}")
            print(f"分类等级: {result.get('rank')}")
            print(f"分类状态: {result.get('taxonomic_status')}")

            # 打印来源数据库信息
            sources = result.get('tax_sources', [])
            print("来源数据库:")
            for source in sources:
                print(f"- {source}")

            # 保存完整结果到文件（可选）
            with open("taxon_info_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print("完整结果已保存到 taxon_info_result.json")

        else:
            # 请求失败，打印错误信息
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")

    except requests.exceptions.RequestException as e:
        # 处理请求异常
        print(f"请求发生异常: {e}")


def download_valid_nwk(ott_ids, output_file="valid_tree.nwk"):
    url = "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree"
    headers = {"Content-Type": "application/json"}
    payload = {"node_ids": [f"ott{id}" for id in ott_ids]}  # 使用node_ids并添加前缀

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        # 解析JSON响应，提取"newick"字段（API可能返回JSON格式的结果）
        response_data = response.json()
        if "newick" in response_data:
            raw_nwk = response_data["newick"]
        else:
            raw_nwk = response.text  # 兜底方案：直接使用响应文本

        # 清理NWK内容：仅保留分号前的有效部分
        cleaned_nwk = raw_nwk.split(';')[0] + ';'  # 取第一个分号前的内容并补充分号

        # 进一步清理可能的首尾空格或特殊字符
        cleaned_nwk = cleaned_nwk.strip().replace('"', '').replace("'", "")

        # 验证基本格式
        if '(' in cleaned_nwk and ')' in cleaned_nwk and cleaned_nwk.endswith(';'):
            with open(output_file, 'w', encoding='ascii') as f:
                f.write(cleaned_nwk)
            print(f"成功提取并保存有效NWK内容至 {output_file}")
            return True
        else:
            print(f"清理后的内容仍无效:\n{cleaned_nwk[:200]}")
            return False

    except requests.exceptions.JSONDecodeError:
        print("错误：API返回非JSON格式内容，尝试直接处理文本")
        return download_fallback_nwk(ott_ids, output_file)  # 调用兜底函数
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def download_fallback_nwk(ott_ids, output_file):
    """兜底方案：直接使用原始文本并强制清理格式"""
    url = "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree"
    headers = {"Content-Type": "application/json"}
    payload = {"node_ids": [f"ott{id}" for id in ott_ids]}

    try:
        response = requests.post(url, headers=headers, json=payload)
        raw_content = response.text

        # 暴力清理：删除所有非NWK字符（括号、分号、字母、数字、冒号、下划线）
        valid_chars = set("(){};_:., \n\t0123456789")
        cleaned_nwk = ''.join([c for c in raw_content if c in valid_chars])

        # 保留最后一个分号前的内容
        if ';' in cleaned_nwk:
            cleaned_nwk = cleaned_nwk.rsplit(';', 1)[0] + ';'

        with open(output_file, 'w', encoding='ascii') as f:
            f.write(cleaned_nwk)
        print(f"兜底方案：已保存清理后的内容至 {output_file}")
        return True
    except Exception as e:
        print(f"兜底方案失败: {e}")
        return False



if __name__ == "__main__":
    #人类770315
    #library(datelife)
#taxa <- c("Homo sapiens", "Mus musculus", "Drosophila melanogaster", "Caenorhabditis elegans", "Danio rerio", "Canis lupus familiaris")
#lin <- get_ott_lineage(taxa)
#print(lin)
    #ott_ids = [770315, 542509, 505714, 395048,1005914,247333]  # 示例ID，需替换为有效OTT ID

    species_ott_ids = [

        {"name": "非洲草原象", "scientific_name": "Loxodonta africana", "ott_id": 541936},
        {"name": "亚洲象", "scientific_name": "Elephas maximus", "ott_id": 541928},
        {"name": "黑猩猩", "scientific_name": "Pan troglodytes", "ott_id": 417950},
        {"name": "山地大猩猩", "scientific_name": "Gorilla beringei", "ott_id": 351685},
        {"name": "北极熊", "scientific_name": "Ursus maritimus", "ott_id": 10732},
        {"name": "大熊猫", "scientific_name": "Ailuropoda melanoleuca", "ott_id": 872573},

        {"name": "苏门答腊虎", "scientific_name": "Panthera tigris sondaica", "ott_id": 445492},
        {"name": "猎豹", "scientific_name": "Acinonyx jubatus", "ott_id": 752759},
        {"name": "非洲野狗", "scientific_name": "Lycaon pictus", "ott_id": 821953},
        {"name": "地中海僧海豹", "scientific_name": "Monachus monachus", "ott_id": 759722},
        {"name": "绿海龟", "scientific_name": "Chelonia mydas", "ott_id": 559133},


        {"name": "玳瑁", "scientific_name": "Eretmochelys imbricata", "ott_id": 430337},
        {"name": "蓝鲸", "scientific_name": "Balaenoptera musculus", "ott_id": 226190},
        {"name": "北大西洋露脊鲸", "scientific_name": "Eubalaena glacialis", "ott_id": 397247},
        {"name": "红毛猩猩", "scientific_name": "Pongo pygmaeus", "ott_id": 770302},


        {"name": "马来貘", "scientific_name": "Tapirus indicus", "ott_id": 93318},
        {"name": "黑足鼬", "scientific_name": "Mustela nigripes", "ott_id": 541439},
        {"name": "伊比利亚猞猁", "scientific_name": "Lynx pardinus", "ott_id": 442049},
        {"name": "加州神鹫", "scientific_name": "Gymnogyps californianus", "ott_id": 316992},
        {"name": "紫蓝金刚鹦鹉", "scientific_name": "Anodorhynchus hyacinthinus", "ott_id": 416115},
        {"name": "非洲灰鹦鹉", "scientific_name": "Psittacus erithacus", "ott_id": 285641},


        {"name": "中华白海豚", "scientific_name": "Sousa chinensis", "ott_id": 187220},
        {"name": "江豚", "scientific_name": "Neophocaena asiaeorientalis", "ott_id": 1019449},
        {"name": "墨西哥钝口螈", "scientific_name": "Ambystoma mexicanum", "ott_id": 984726},
        {"name": "欧洲鳗鲡", "scientific_name": "Anguilla anguilla", "ott_id": 854201},


        {"name": "比熊犬", "scientific_name": "Canis lupus familiaris", "ott_id":  247333},


        {"name": "黑狼", "scientific_name": "Canis lupus", "ott_id": 247333},
        {"name": "乌骨鸡", "scientific_name": "Gallus gallus domesticus", "ott_id": 547470},
        {"name": "黑色长颈鹿", "scientific_name": "Giraffa camelopardalis", "ott_id": 768674},
        {"name": "黑色美洲豹", "scientific_name": "Panthera onca", "ott_id": 42322},
        {"name": "黑色海豹", "scientific_name": "Phoca vitulina", "ott_id": 698422},
        {"name": "黑色松鼠", "scientific_name": "Sciurus carolinensis", "ott_id": 410925},
        {"name": "黑色大象", "scientific_name": "Loxodonta africana", "ott_id": 541936},
        {"name": "黑色斑马", "scientific_name": "Equus quagga", "ott_id": 124776},
        {"name": "黑蛇", "scientific_name": "Pantherophis obsoletus", "ott_id": 530777},
        {"name": "黑色鳄鱼", "scientific_name": "Crocodylus acutus", "ott_id": 870591},
        {"name": "黑鹿", "scientific_name": "Rusa unicolor", "ott_id": 844145},
        {"name": "犏牛", "scientific_name": "Bos grunniens × Bos taurus", "ott_id": 381164},
        {"name": "黑色火烈鸟", "scientific_name": "Phoenicopterus roseus", "ott_id": 595567},
        {"name": "黑色皇企鹅", "scientific_name": "Aptenodytes forsteri", "ott_id": 494370},
        {"name": "黑色飞蛾", "scientific_name": "Biston betularia", "ott_id": 968114},

        # 新增的模式生物（补充部分） ott_ids = [770315, 542509, 505714, 395048,1005914,247333]  # 示例ID，需替换为有效OTT ID
        {"name": "人类", "scientific_name": "Homo sapiens", "ott_id": 770315},
        {"name": "小鼠", "scientific_name": "Mus musculus", "ott_id": 542509},
        {"name": "大鼠", "scientific_name": "Rattus norvegicus", "ott_id": 271555},
        {"name": "果蝇", "scientific_name": "Drosophila melanogaster", "ott_id": 505714},
        {"name": "线虫", "scientific_name": "Caenorhabditis elegans", "ott_id": 395048},
        {"name": "海鞘", "scientific_name": "Ciona intestinalis", "ott_id": 247333},
        {"name": "斑马鱼", "scientific_name": "Danio rerio", "ott_id": 1005914},
    ]

    # 初始化ott_ids数组
    ott_ids = []

    # 循环提取ott_id
    for species in species_ott_ids:
        ott_id = species.get('ott_id')
        if ott_id is not None:  # 过滤掉可能存在的None值
            ott_ids.append(ott_id)

    # 打印结果（可选）
    print("提取的OTT IDs:", ott_ids)
    if download_valid_nwk(ott_ids):
        print("请使用可视化工具检查文件是否可解析")
    #
    # for ott_id in ott_ids:
    #     get_taxon_info(ott_id)

