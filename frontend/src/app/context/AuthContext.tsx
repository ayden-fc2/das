"use client";

import { createContext, useContext, useEffect, useState } from "react";
import {getAuthRole, getUserInfoApi} from "@/app/api/account";
import {ROLE_TYPE} from "@/app/types/common";

// 定义类型
interface AuthContextType {
    role: string | null;
    isSuperManager: boolean;
    basicUserInfo: BasicUserInfo | null;
    setRole: (role: string | null) => void;
    refreshRole: () => void;
}

interface BasicUserInfo {
    nickname: string | null;
    avatar: string | null;
    email: string | null;
}

// 创建 Context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [role, setRole] = useState<string | null>(null);
    const [basicUserInfo, setBasicUserInfo] = useState<BasicUserInfo | null>(null);
    const [token, setToken] = useState<string | null>(null); // 监听 Token
    const [isSuperManager, setIsSuperManager] = useState<boolean>(false);

    useEffect(() => {
        // 获取 Token
        const storedToken = localStorage.getItem("jwt");
        if (storedToken) {
            setToken(storedToken);
        }
    }, []);

    // 应用启动时获取用户角色
    useEffect(() => {
        if (token) {
            handleTokenChange()
        }
    }, [token]);

    const refreshRole = () => {
       handleTokenChange()
    }

    // 处理token更新
    const handleTokenChange = () => {
        getAuthRole().then(res=>{
            setRole(res.data)
            const isSuperManager = Object.values(res.data).some((orgRoles: any) =>
                orgRoles.some((role: any) => role === ROLE_TYPE.S_MANAGER)
            );
            setIsSuperManager(isSuperManager);
        }).catch(err=>{
            console.log(err, '获取用户角色失败')
        })
        getUserInfoApi().then(res=>{
            if (res.success) {
                setBasicUserInfo({
                    avatar: res.data.avatar,
                    nickname: res.data.nickName,
                    email: res.data.email,
                })
                return
            }
            throw new Error('获取用户信息失败')
        }).catch(err=>{
            console.log(err)
        })
    }


    return (
        <AuthContext.Provider value={{ role, isSuperManager, basicUserInfo, setRole, refreshRole }}>
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
