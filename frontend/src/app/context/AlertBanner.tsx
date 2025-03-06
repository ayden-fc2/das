"use client"

import React, {createContext, useContext, useEffect, useState} from "react";
import { Snackbar, Alert, AlertColor } from "@mui/material";
import {info, setAlerter} from "@/app/utils/alerter";

// 定义 Alert 上下文的类型
interface AlertContextProps {
    showAlert: (message: string, severity: AlertColor) => void;
}

// 创建上下文
const AlertContext = createContext<AlertContextProps | undefined>(undefined);

// 提供者组件
export const AlertProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [open, setOpen] = useState(false);
    const [message, setMessage] = useState("");
    const [severity, setSeverity] = useState<AlertColor>("info");

    const showAlert = (msg: string, sev: AlertColor) => {
        setMessage(msg);
        setSeverity(sev);
        setOpen(true);
    };

    useEffect(() => {
        setAlerter({ showAlert });
        // 全局提示，observer账号
        info('The login and registration functionality is currently unavailable. Please use the test account "1234@test.com" with the password "1234".')
    }, []);

    const handleClose = () => {
        setOpen(false);
    };

    return (
        <AlertContext.Provider value={{ showAlert }}>
            {children}
            <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
                <Alert onClose={handleClose} severity={severity}>
                    {message}
                </Alert>
            </Snackbar>
        </AlertContext.Provider>
    );
};

// 自定义 Hook 用于消费 AlertContext
export const useAlert = (): AlertContextProps => {
    const context = useContext(AlertContext);
    if (!context) {
        throw new Error("useAlert must be used within an AlertProvider");
    }
    return context;
};
