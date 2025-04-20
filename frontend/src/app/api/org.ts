import {get, post} from "@/app/utils/api";

const urls = {
    CREATE_ORG: '/auth-service/org/create', // 新建群组
    JOIN_ORG: '/auth-service/org/join', // 加入群组
    GET_ORG: '/auth-service/org/getMyOrgs', // 获取用户群组列表
    GET_ALL_ORG: '/auth-service/org/getAllMyOrgs', // 不分页查询
    ORG_NUM: '/auth-service/org/getMyOrgsNum', // 获取用户群组总数
    QUIT_ORG: '/auth-service/org/quit', // 用户退出群组
    UPDATE_ORG: '/auth-service/org/update', // 群组基本信息
    UPDATE_ORG_CODE: '/auth-service/org/updateCode', // 更新组织代码
    DELETE_ORG: '/auth-service/org/delete', // 删除组织
    GET_ORG_MEMBERS: '/auth-service/org/getOrgsMember', // 获取组织内所有用户
    DELETE_ORG_MEMBER: '/auth-service/org/deleteUser', // 删除组织内用户
    MANAGE_ORG_USER_ROLE: '/auth-service/org/manageUserRoles', // 管理用户角色
}

export const createOrg = (orgName: string, orgDesc: string) => {
    return get(urls.CREATE_ORG, {
        orgName,
        orgDesc
    })
}

export const joinOrg = (orgCode: string) => {
    return get(urls.JOIN_ORG, {
        orgCode
    })
}

export const getMyOrgs = (page: number, size: number) => {
    return get(urls.GET_ORG, {
        page,
        size
    })
}

export const getAllMyOrgs = () => {
    return get(urls.GET_ALL_ORG)
}

export const getMyOrgNum = () => {
    return get(urls.ORG_NUM)
}

export const quitOrgByUsr = (orgId: number) => {
    return get(urls.QUIT_ORG, {
        orgId
    })
}

export const updateOrgInfo = (orgId: number, orgName: string, orgDesc: string) => {
    return get(urls.UPDATE_ORG, {
        orgId,
        orgName,
        orgDesc
    })
}

export const updateOrgCode = (orgId: number) => {
    return get(urls.UPDATE_ORG_CODE, {
        orgId,
    })
}

export const deleteOrg = (orgId: number) => {
    return get(urls.DELETE_ORG, {
        orgId
    })
}

export const getOrgMembers = (orgId: number) => {
    return get(urls.GET_ORG_MEMBERS, {
        orgId
    })
}

export const deleteOrgMember = (orgId: number, userId: number) => {
    return get(urls.DELETE_ORG_MEMBER, {
        orgId,
        userId
    })
}

export const updateOrgMemberRole = (orgId: number, userId: number, roleIds: string) => {
    return get(urls.MANAGE_ORG_USER_ROLE, {
        orgId,
        userId,
        roleIds
    })
}