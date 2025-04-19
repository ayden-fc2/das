package com.auth.oauth2.mapper;

import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.entity.RelationshipSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface RelationshipMapper {
    @Insert("INSERT INTO relationship_st (accountId, orgId, authorityId) VALUES (#{userId}, #{orgId}, #{i})")
    void createRelationship(@Param("userId") int userId, @Param("orgId") int orgId, @Param("i") int i);

    @Delete("DELETE FROM relationship_st WHERE orgId = #{orgId}")
    boolean deleteRelationshipByOrgId( @Param("orgId") int orgId);

    @Delete("DELETE FROM relationship_st WHERE accountId = #{userId} AND orgId = #{orgId}")
    boolean quitOrg( @Param("userId") int userId, @Param("orgId") int orgId);

    @Select("SELECT a.accountId, a.nickName, a.avatar, a.phoneNum, " +
            "GROUP_CONCAT(r.authorityId ORDER BY r.authorityId) AS authorityIds " +
            "FROM relationship_st r " +
            "JOIN account_st a ON r.accountId = a.accountId " +
            "WHERE r.orgId = #{orgId} " +
            "GROUP BY a.accountId " +
            "LIMIT #{size} OFFSET #{offset}")
    List<JSONObject> getOrgsMember(@Param("orgId") int orgId,
                                   @Param("offset") int offset,
                                   @Param("size") int size);


    @Select("SELECT o.org_id, o.org_name, o.created_time, o.creater_id, o.org_desc, o.org_code, " +
            "GROUP_CONCAT(r.authorityId ORDER BY r.authorityId) AS authorityIds, " +  // 注意这里加了逗号
            "a.nickName, a.avatar, a.phoneNum " +
            "FROM relationship_st r " +
            "JOIN org_st o ON r.orgId = o.org_id " +
            "JOIN account_st a ON o.creater_id = a.accountId " +
            "WHERE r.accountId = #{userId} " +
            "GROUP BY o.org_id " +
            "LIMIT #{size} OFFSET #{offset}")
    List<JSONObject> getMyOrgs(@Param("userId") int userId,
                               @Param("offset") int offset,
                               @Param("size") int size);

    @Select("SELECT COUNT(DISTINCT o.org_id) " +
            "FROM relationship_st r " +
            "JOIN org_st o ON r.orgId = o.org_id " +
            "JOIN account_st a ON o.creater_id = a.accountId " +
            "WHERE r.accountId = #{userId}")
    Integer getMyOrgsNum(@Param("userId") int userId);


    @Delete("DELETE FROM relationship_st WHERE accountId = #{userId} AND orgId = #{orgId}")
    boolean deleteUser(int managerId, @Param("orgId") int orgId, @Param("userId") int userId);

    @Select("SELECT * FROM relationship_st WHERE accountId = #{userId} AND orgId = #{orgId}")
    List<RelationshipSt> getRelationship( @Param("userId") int userId, @Param("orgId") int orgId);
}
