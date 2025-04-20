import {get, post} from "@/app/utils/api";

const urls = {
    CREATE_PROJECT: '/dwg-handler/project/addProjectByOrgId', // 新增项目
    GET_PROJECTS: '/dwg-handler/project/getProjectsByOrgId', // 获取项目
}

export const createProjectApi = (orgId: number, title: string, description: string) => {
    return get(urls.CREATE_PROJECT, {orgId, title, description})
}

export const getAllProjectsApi = (orgId: number) => {
    return get(urls.GET_PROJECTS, {orgId})
}