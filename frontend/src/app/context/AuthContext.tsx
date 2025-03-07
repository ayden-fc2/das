"use client";

import { createContext, useContext, useEffect, useState } from "react";
import {getAuthRole} from "@/app/api/das";

// 定义类型
interface AuthContextType {
    role: string | null;
    setRole: (role: string | null) => void;
    refreshRole: () => void;
}

// 创建 Context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [role, setRole] = useState<string | null>(null);

    const [token, setToken] = useState<string | null>(null); // 监听 Token

    useEffect(() => {
        // 获取 Token
        const storedToken = localStorage.getItem("token");
        if (storedToken) {
            setToken(storedToken);
        }
    }, []);

    // 应用启动时获取用户角色
    useEffect(() => {
        if (token) {
            getAuthRole().then(res=>{
                setRole(res.data)
            }).catch(err=>{
                console.log(err, '获取用户角色失败')
            })
        }
    }, [token]);

    const refreshRole = () => {
        getAuthRole()
            .then((res) => {
                setRole(res.data);
            })
            .catch((err) => {
                console.log(err, "刷新用户角色失败");
            });
    }


    return (
        <AuthContext.Provider value={{ role, setRole, refreshRole }}>
            {children}
        </AuthContext.Provider>
    );
}

// 自定义 Hook 获取角色
export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
