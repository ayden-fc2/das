// utils/api.ts

import axios from "axios";
import jwt from 'jwt-simple';
import {navigateTo} from "@/app/utils/navigator";
import {warning} from "@/app/utils/alerter";
import {MyResponse} from "@/app/types/common";

// 创建 axios 实例
const api = axios.create({
    baseURL: "http://www.fivecheers.com:2073", // API 根路径
    timeout: 10000, // 请求超时时间
});

// 请求拦截器：在每次请求前设置 Authorization 头
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("jwt"); // 从 localStorage 获取 JWT
        if (token && !isTokenExpired(token)) {
            config.headers["Authorization"] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 响应拦截器：处理全局错误
api.interceptors.response.use(
    (response) => response, // 正常响应直接返回
    (error) => {
        const { response } = error;
        if (response?.status === 401) {
            localStorage.removeItem("jwt"); // 清除无效 Token并调转回登陆页面
            warning("Token expired, redirecting to Login Page...");
            navigateTo("/pages/sign/login")
        } else if (response?.status >= 500) {
            console.error("Server error:", response?.statusText || "Unknown error");
            // 其他全局错误逻辑，比如展示通知
        }
        return Promise.reject(error); // 将错误继续抛出
    }
);

// 通用 GET 方法封装
export const get = async (url: string, params?: object): Promise<MyResponse> => {
    try {
        const response = await api.get<MyResponse>(url, { params });
        return response.data;
    } catch (error) {
        throw error; // 将错误抛出给调用方处理
    }
};

// 通用 POST 方法封装
export const post = async (url: string, data: object): Promise<MyResponse> => {
    try {
        const response = await api.post<MyResponse>(url, data);
        return response.data;
    } catch (error) {
        throw error; // 将错误抛出给调用方处理
    }
};

export const isTokenExpired = (token: string): boolean => {
    try {
        const decoded = jwt.decode(token, '', true); // `true` 表示不验证签名
        if (!decoded.exp) return true; // 如果没有 exp 字段，认为已过期
        const now = Math.floor(Date.now() / 1000);
        return decoded.exp < now;
    } catch (err) {
        console.error('Invalid token:', err);
        return true; // 解析失败，认为已过期
    }
};

export default api;
