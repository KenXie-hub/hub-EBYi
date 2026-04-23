import sqlite3
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text # ORM 框架
import pandas as pd
import requests
import time
import jwt


# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db') # 数据库文件，包含多张表

# 创建游标对象
cursor = conn.cursor()

class DBParser:
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        mysql: mysql://root:123456@localhost:3306/mydb?charset=utf8mb4
        sqlite: sqlite://chinook.db
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names() # 获取table信息

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                # 获取当前字段
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计most frequency value
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]


    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]
    
    def getTableNums(self):
        '''
        数据库中有多少张表
        '''
        return len(self.table_names)
    
    def getEmployeeCount(self):
        '''
        员工表中有多少条记录
        '''
        # 获取元数据
        metaData = MetaData()
        metaData.reflect(bind=self.engine)

        # 拿到员工表
        employees_table = metaData.tables['employees']

        # ORM查询总数
        query = select(func.count()).select_from(employees_table)
        result = self.conn.execute(query)
        return result.scalar()
    def userCount(self):
        '''
        在数据库中所有客户个数和员工个数分别是多少
        customers
        employees
        '''
        # 获取元数据
        metaData = MetaData()
        metaData.reflect(bind=self.engine)

        customers_table = metaData.tables.get('customers')
        employees_table = metaData.tables.get('employees')

        cust_count = 0
        emp_count = 0
        if customers_table is not None:
            query = select(func.count()).select_from(customers_table)
            cust_count = self.conn.execute(query).scalar()
        if employees_table is not None:
            query = select(func.count()).select_from(employees_table)
            emp_count = self.conn.execute(query).scalar()

        return {'customers': cust_count, 'employees': emp_count}


    def execute_sql(self, sql):
        '''
        运行SQL
        '''
        result = self.conn.execute(text(sql))
        return list(result)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]
    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

parser = DBParser('sqlite:///./chinook.db')
parser.get_data_relations()
# print(parser.userCount())




def ask_glm(question):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token(".Qfo5i1kHjGsMOZMY", 1000)
    }
    data = {
        "model": "glm-3-turbo",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    response = requests.post(url, headers=headers, json=data, timeout=10)
    return response.json()

def normalize_sql_for_sqlite(sql: str) -> str:
    """
    简单把常见的 information_schema 查询转换为 sqlite_master 的等价查询。
    仅处理典型的“表计数/列出表名”场景。
    """
    lsql = sql.lower()
    if 'information_schema.tables' in lsql:
        if 'count(' in lsql:
            return "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
        else:
            return "SELECT name FROM sqlite_master WHERE type='table';"
    return sql

def nl2sql_agent(question: str):
    # 优先处理常见问题，直接用 parser 查询，避免调用大模型
    # q = question.lower()
    # if '多少张表' in q or '多少表' in q:
    #     return f"数据库中总共有 {parser.getTableNums()} 张表。"
    # if '员工表' in q and ('多少' in q or '多少条' in q):
    #     return f"员工表中有 {parser.getEmployeeCount()} 条记录。"
    # if ('客户' in q and '员工' in q) and ('分别' in q or '各' in q):
    #     counts = parser.userCount()
    #     return f"客户总数为 {counts['customers']}，员工总数为 {counts['employees']}。"

    # 未命中上面规则则回退到原始的 NL->SQL 流程
    # 获取数据库结构
    table_names = parser.table_names
    table_info = ""
    for tbl in table_names:
        sample = parser.get_table_sample(tbl).to_markdown()
        schema = parser.get_table_fields(tbl).to_markdown()
        table_info += f"表名：{tbl}\n样例：{sample}\n字段：{schema}\n\n"
    
    # 给大模型发送：数据库结构 + 问题 -> 生成SQL（明确说明 SQLite）
    prompt = f"""
    你是数据库专家。目标数据库：SQLite。请只输出 SQLite 兼容的 SQL（不要任何解释或其它文本）。
    数据库表结构：
    {table_info}
    用户问题：{question}
    """
    # 调大模型生成SQL
    res = ask_glm(prompt)
    sql = res['choices'][0]['message']['content'].strip('`').replace('sql\n', '').strip()
    print(f"生成SQL：{sql}")

    # 尝试执行SQL，若出现与 information_schema 相关的错误则转换后重试
    try:
        result = parser.execute_sql(sql)
    except Exception as e:
        err = str(e).lower()
        print("执行SQL出错：", err)
        # 情况1：模型返回了多条语句（例如两个 SELECT） -> 按分号拆开逐条执行
        if 'you can only execute one statement' in err or 'one statement at a time' in err or ';' in sql:
            stmts = [s.strip() for s in sql.split(';') if s.strip()]
            combined = []
            for stmt in stmts:
                try:
                    r = parser.execute_sql(stmt)
                    combined.append(r)
                except Exception as e2:
                    combined.append({'error': str(e2), 'sql': stmt})
            result = combined
            sql = " ; ".join(stmts)
        # 情况2：information_schema（Postgres 风格） -> 转换为 sqlite_master 再试
        elif 'information_schema' in sql.lower() or 'information_schema' in err:
            alt_sql = normalize_sql_for_sqlite(sql)
            print("尝试将 SQL 转为 SQLite 版本并重试：", alt_sql)
            try:
                result = parser.execute_sql(alt_sql)
                sql = alt_sql  # 更新为实际执行的 SQL
            except Exception as e2:
                print("重试仍失败：", str(e2))
                return f"执行 SQL 失败：{e2}\n原始 SQL：{sql}\n尝试的替代 SQL：{alt_sql}"
        else:
            return f"执行 SQL 出错：{e}\n原始 SQL：{sql}"
    print(f"查询结果：{result}")
    answer_rewrite_prompt = f'''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

    原始问题：{question}

    执行SQL：{sql}

    原始结果：{result}
    '''
    nl_answer = ask_glm(answer_rewrite_prompt)['choices'][0]['message']['content']
    return nl_answer


# count = parser.getEmployeeCount()
# print(count, '查看所有的count的值')

if __name__ == '__main__':
    q1 = "数据库中总共有多少张表"
    q2 = "员工表中有多少条记录"
    q3 = "在数据库中所有客户个数和员工个数分别是多少"
    
    # print("\n【问题1】", q1)
    # print("【回答】", nl2sql_agent(q1))

    # print("\n【问题2】", q2)
    # print("【回答】", nl2sql_agent(q2))

    print("\n【问题3】", q3)
    print("【回答】", nl2sql_agent(q3))
