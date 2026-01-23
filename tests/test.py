import asyncio


# class AsyncIterator:
#     def __init__(self):
#         self.count = 0


#     def __aiter__(self):
#         return self


#     async def __anext__(self):
#         if self.count < 5:
#             self.count += 1
#             return self.count
#         else:
#             raise StopAsyncIteration


# async def async_for_example():
#     async for number in AsyncIterator():
#         print(number)


# asyncio.run(async_for_example())

import asyncio
import aiohttp

async def fetch_url(session, url):
    """异步请求单个URL"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            # 获取响应状态码和内容
            status = response.status
            content = await response.text()  # 文本内容（response.json()获取JSON）
            return {
                "url": url,
                "status": status,
                "content_length": len(content)
            }
    except Exception as e:
        return {
            "url": url,
            "error": str(e)
        }

async def main():
    # 创建异步HTTP会话（复用连接，提升效率）
    async with aiohttp.ClientSession() as session:
        result = await fetch_url(session, "https://www.baidu.com")
        print(f"请求结果：{result}")

asyncio.run(main())


