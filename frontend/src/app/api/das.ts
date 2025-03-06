import {get} from "@/app/utils/api";

const urls = {
    GET_PUBLIC_PROJECTS: '/dwg-handler/read/getPublicList', // demo获取所有公共项目
    GET_AUTH_ROLE: '/auth-service/getRoles', // 获取用户角色
    GEN_PUBLIC_PROJECT: '/dwg-handler/cop/genAnalysis'
}

export const getPublicList = ()=> {
    return get(urls.GET_PUBLIC_PROJECTS)
}

export const getAuthRole = () => {
    return get(urls.GET_AUTH_ROLE)
}

export const analysisPublicProject = (
                                            projectName: string,
                                            dwgPath: string,
                                            isPublic: number
                                      ) => {
    return get(urls.GEN_PUBLIC_PROJECT, {
        projectName,
        dwgPath,
        isPublic,
    })
}


