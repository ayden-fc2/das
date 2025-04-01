import {get, post} from "@/app/utils/api";

const urls = {
    LOGIN_API: "/auth-service/sign/signInCheck", // 登陆
    GET_AUTH_ROLE: '/auth-service/getRoles', // 获取用户角色
    GET_CODE: '/auth-service/sign/getPhoneCode', // 获取验证码
    REGISTER_API: '/auth-service/sign/signUp', // 注册
    RESET_PASSWORD_API: '/auth-service/sign/resetPassword' // 重设密码
}

export const loginApi = (email: string, password: string) => {
    return get(urls.LOGIN_API, {email, password})
}

export const getAuthRole = () => {
    return get(urls.GET_AUTH_ROLE)
}

// mode = 0 注册； mode = 1 重设密码
export const getCode = (email: string, mode: number) => {
    return get(urls.GET_CODE, {email, mode})
}

export const registerApi = (email: string, password: string, code: string | number, name: string) => {
    return get(urls.REGISTER_API, {
        email,
        password,
        mode: 0,
        code,
        name
    })
}

export const resetPasswordApi = (email: string, newPassword: string, code: string | number) => {
    return get(urls.RESET_PASSWORD_API, {
        email,
        newPassword,
        code,
        mode: 1
    })
}