package com.auth.oauth2.mapper;

import com.auth.oauth2.entity.AccountSt;
import com.auth.oauth2.entity.RelationshipSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UserMapper {

    @Select("SELECT * FROM account_st WHERE phoneNum = #{phoneNum}")
    List<AccountSt> selectAccounts(String phoneNum);


    @Select("SELECT * FROM relationship_st WHERE accountId = #{userId}")
    List<RelationshipSt> selectRelations(long userId);

    @Insert("INSERT INTO account_st(phoneNum, passwordDetail) VALUES(#{phoneNum}, #{passwordDetail})")
    @Options(useGeneratedKeys = true, keyProperty = "accountId", keyColumn = "accountId")
    void insertNewUser(AccountSt accountSt);

    @Insert("INSERT INTO relationship_st(accountId, authorityId) VALUES(#{accountId}, #{authorityId})")
    void insertRelationship(long accountId, long authorityId);

    @Update("UPDATE account_st SET passwordDetail = #{newPassword} WHERE phoneNum = #(phoneNum)")
    void updatePassword(String phoneNum, String newPassword);
}
