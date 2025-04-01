import {MyResponse} from "@/app/types/common";
import {loginApi} from "@/app/api/account";
import {err, success} from "@/app/utils/alerter";
import {navigateTo} from "@/app/utils/navigator";

export const handleLoginProcess = async (email: string, password: string, authContext: any) => {
    console.log("Logging in with:", {email, password});
    try {
        const response: MyResponse = await loginApi(email, password)
        if (response.success === 1 && response.data) {
            localStorage.setItem("jwt", response.data);
            authContext.refreshRole()
            success("Login successful, your login status will be saved。")
            navigateTo("/pages/dashboard")
        } else {
            console.error("Login failed:", response.message);
            err(response.message)
        }
    } catch (error) {
        console.error("Login Error:", error);
        err("An issue has occurred. Please try again.")
    }
}