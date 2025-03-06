"use client";

import { createContext, useContext, useEffect, useState } from "react";
import {getAuthRole} from "@/app/api/das";

// 定义类型
interface AuthContextType {
    role: string | null;
    setRole: (role: string | null) => void;
}

// 创建 Context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [role, setRole] = useState<string | null>(null);

    // 应用启动时获取用户角色
    useEffect(() => {
        getAuthRole().then(res=>{
            setRole(res.data)
        }).catch(err=>{
            console.log(err, '获取用户角色失败')
        })
    }, []);

    return (
        <AuthContext.Provider value={{ role, setRole }}>
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
