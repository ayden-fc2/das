export interface Project {
    projectName: string;
    createdTime?: string;
    isPublic: boolean;
    dwgPath: string;
    jsonPath?: string;
    analysised: number;
    id: number;
}