import asyncio
import time


candidates = []


async def with_lock(lock):
    async with lock:
        print("Waiting 3 secs...", (candidates))
        await asyncio.sleep(3)
        print("after 3 secs", candidates)


async def add_items(lock):
    async with lock:
        print(candidates)
        await asyncio.sleep(0.5)
        candidates.append(100)
        print(candidates)


async def main():
    lock = asyncio.Lock()
    await asyncio.gather(
        with_lock(lock),
        add_items(lock),
    )


asyncio.run(main())
