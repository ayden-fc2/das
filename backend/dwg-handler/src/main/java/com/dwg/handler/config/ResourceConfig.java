package com.dwg.handler.config;

import com.example.common.enums.UserType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;
import org.springframework.security.oauth2.config.annotation.web.configuration.ResourceServerConfigurerAdapter;
import org.springframework.security.oauth2.config.annotation.web.configurers.ResourceServerSecurityConfigurer;
import org.springframework.security.oauth2.provider.token.TokenStore;

@Configuration
@EnableResourceServer
public class ResourceConfig extends ResourceServerConfigurerAdapter {
    @Override
    public void configure(HttpSecurity http) throws Exception {
        http
                .authorizeRequests()
                .antMatchers("/read").hasAnyAuthority(
                        UserType.Observer.getType() // 观察者可以访问 /read 接口
                        )
                .antMatchers("/cop/genAnalysis", "/cop/genAnalysisOverview").hasAnyAuthority(
                        UserType.SuperManager.getType() // 超级管理员才可以操作
                )
                .anyRequest().authenticated()
                .and()
                .formLogin().disable()
                .csrf().disable(); // 禁用 CSRF，可能需要根据实际情况启用
    }

    @Override
    public void configure(ResourceServerSecurityConfigurer resources) throws Exception {
        resources.tokenStore(jwtTokenStore);
    }

    @Autowired
    TokenStore jwtTokenStore;
}
