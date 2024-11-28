import { useRouter } from "next/navigation";
import { useSearchParams } from "next/navigation";

let router: ReturnType<typeof useRouter> | null = null;

// 提供一个初始化 router 的方法
export const setRouter = (r: ReturnType<typeof useRouter>) => {
    router = r;
};

// 封装一个跳转函数
export const navigateTo = (path: string, params?: Record<string, string | number>) => {
    if (router) {
        const queryString = params
            ? "?" + new URLSearchParams(params as Record<string, string>).toString()
            : "";
        router.push(path + queryString);
    } else {
        console.error("Router is not initialized.");
    }
};

// 提供一个方法解析目标页面的查询参数
export const useQueryParams = (): Record<string, string> => {
    const searchParams = useSearchParams();
    const params: Record<string, string> = {};
    if (searchParams) {
        searchParams.forEach((value, key) => {
            params[key] = value;
        });
    }
    return params;
};