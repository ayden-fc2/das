export interface MyResponse {
    success: boolean;
    message: string;
    data?: any;
}

export enum ROLE_TYPE {
    S_MANAGER= "SUPER_MANAGER_TYPE",
    MANAGER= "MANAGER_TYPE",
    CONTROLLER= "CONTROLLER_TYPE",
    OBSERVER= "OBSERVER_TYPE",
}