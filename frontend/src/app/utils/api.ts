// utils/api.ts

import axios from "axios";

// 创建 axios 实例
const api = axios.create({
    baseURL: "http://www.fivecheers.com:2073", // API 根路径
    timeout: 10000, // 请求超时时间
});

// 请求拦截器：在每次请求前设置 Authorization 头
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("jwt"); // 从 localStorage 获取 JWT
        if (token) {
            config.headers["Authorization"] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 通用 GET 方法封装
export const get = async <T>(url: string, params?: object): Promise<T> => {
    try {
        const response = await api.get<T>(url, { params });
        return response.data;
    } catch (error) {
        console.error("API GET Error:", error);
        throw error; // 将错误抛出给调用方处理
    }
};

// 通用 POST 方法封装
export const post = async <T>(url: string, data: object): Promise<T> => {
    try {
        const response = await api.post<T>(url, data);
        return response.data;
    } catch (error) {
        console.error("API POST Error:", error);
        throw error; // 将错误抛出给调用方处理
    }
};

export default api;
