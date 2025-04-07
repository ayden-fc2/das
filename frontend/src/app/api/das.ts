import {get, post} from "@/app/utils/api";

const urls = {
    GET_PUBLIC_PROJECTS: '/dwg-handler/read/getPublicList', // demo获取所有公共项目
    GEN_PUBLIC_PROJECT: '/dwg-handler/cop/genAnalysis', // 获取公共项目分析结果 - LibreDWG
    GEN_PUBLIC_PROJECT_GRAPHML: '/dwg-handler/cop/genAnalysisOverview', // 二次分析 - GraphML
    GET_PUBLIC_PROJECT_COMPONENTS: '/dwg-handler/read/getProjectGraph', // 获取分析结果-组件
    GET_PUBLIC_PROJECT_GRAPH: '/dwg-handler/read/getProjectGraphStructure', // 获取分析结果-图结构
    GEN_TRACE: '/dwg-handler/cop/genTrace',
}

export const getPublicList = ()=> {
    return get(urls.GET_PUBLIC_PROJECTS)
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

// 获取项目分析结果 - 组件
export const getProjectComponents = (projectId: number) => {
    return get(urls.GET_PUBLIC_PROJECT_COMPONENTS, {
        projectId,
    })
}

// 获取项目分析结果 - 图结构
export const getProjectGraph = (projectId: number) => {
    return get(urls.GET_PUBLIC_PROJECT_GRAPH, {
        projectId,
    })
}

// 故障预测
export const genTraceApi = (projectId: number, faultIds: any) => {
    return get(urls.GEN_TRACE, {
        projectId,
        faultIds
    })
}

