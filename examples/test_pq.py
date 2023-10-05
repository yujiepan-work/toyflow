import asyncio
import random




async def main():
    q = asyncio.PriorityQueue()
    for i in range(4):
        q.put_nowait(PrioritizedItem([i, random.randint(0, 111)], str(i)))

    for _ in range(4):
        print(await q.get())


asyncio.run(main())
