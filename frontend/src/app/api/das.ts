import {get, post} from "@/app/utils/api";

const urls = {
    GET_PUBLIC_PROJECTS: '/dwg-handler/read/getPublicList', // demo获取所有公共项目
    GET_AUTH_ROLE: '/auth-service/getRoles', // 获取用户角色
    GEN_PUBLIC_PROJECT: '/dwg-handler/cop/genAnalysis', // 获取公共项目分析结果 - LibreDWG
    GEN_PUBLIC_PROJECT_GRAPHML: '/dwg-handler/cop/genAnalysisOverview', // 获取公共项目分析结果 - GraphML
    GET_PUBLIC_PROJECT_GRAPHML: '/dwg-handler/read/getProjectGraph'
}

export const getPublicList = ()=> {
    return get(urls.GET_PUBLIC_PROJECTS)
}

export const getAuthRole = () => {
    return get(urls.GET_AUTH_ROLE)
}

// 分析公共项目 step1 - LibreDWG
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

// 分析公共项目 step2 - GraphML
export const analysisPublicProjectGraphML = (postData: any) => {
    return post(urls.GEN_PUBLIC_PROJECT_GRAPHML, postData)
}

// 获取公共项目分析结果 - GraphML
export const getProjectGraph = (projectId: number) => {
    return get(urls.GET_PUBLIC_PROJECT_GRAPHML, {
        projectId,
    })
}


