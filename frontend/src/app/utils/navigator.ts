import { useRouter } from "next/navigation";

let router: ReturnType<typeof useRouter> | null = null;

// 提供一个初始化 router 的方法
export const setRouter = (r: ReturnType<typeof useRouter>) => {
    router = r;
};

// 封装一个跳转函数
export const navigateTo = (path: string) => {
    if (router) {
        router.push(path);
    } else {
        console.error("Router is not initialized.");
    }
};
