import {get, post} from "@/app/utils/api";

const urls = {
    CREATE_PROJECT: '/dwg-handler/project/addProjectByOrgId', // 新增项目
    GET_PROJECTS: '/dwg-handler/project/getProjectsByOrgId', // 获取项目
    CREATE_CHILD_PROJECT: '/dwg-handler/project/addChildProjectByProjectKey', // 增加子项目
    DELETE_PROJECT: '/dwg-handler/project/deleteProjectByProjectKey', // 删除项目
    UPLOAD_DCS: '/dwg-handler/cop/uploadDwgByOrgId', // 上传DCS项目
}

export const createProjectApi = (orgId: number, title: string, description: string) => {
    return get(urls.CREATE_PROJECT, {orgId, title, description})
}

export const getAllProjectsApi = (orgId: number) => {
    return get(urls.GET_PROJECTS, {orgId})
}

export const createChildProjectApi = (orgId: number, projectKey: number, title: string, description: string) => {
    return get(urls.CREATE_CHILD_PROJECT, {orgId, projectKey, title, description})
}

export const deleteProjectApi = (orgId: number, projectKey: number) => {
    return get(urls.DELETE_PROJECT, {
        orgId,
        projectKey
    })
}

export const updateProjectApi = (dwgPath: string, orgId: number, projectId: number) => {
    return get(urls.UPLOAD_DCS, {dwgPath ,orgId, projectId})
}