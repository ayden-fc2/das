import { useAlert } from "@/app/components/AlertBanner";

let alerter: ReturnType<typeof useAlert> | null = null;

// 提供一个初始化 router 的方法
export const setAlerter = (r: ReturnType<typeof useAlert>) => {
    alerter = r;
};

// 封装info, warnings, error, success
export const info = (message: string) => {
    if (alerter) {
        alerter.showAlert(message, "info");
    } else {
        console.error("Alerter is not initialized.");
    }
};

export const warning = (message: string) => {
    if (alerter) {
        alerter.showAlert(message, "warning")
    } else {
        console.error("Alerter is not initialized.");
    }
};

export const err = (message: string) => {
    if (alerter) {
        alerter.showAlert(message, "error");
    } else {
        console.error("Alerter is not initialized.");
    }
};

export const success = (message: string) => {
    if (alerter) {
        alerter.showAlert(message, "success");
    } else {
        console.error("Alerter is not initialized.");
    }
};