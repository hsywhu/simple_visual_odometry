#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef CONFIG_H
#define CONFIG_H

// std
#include <opencv2/core/core.hpp>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>

using namespace std;

namespace myslam 
{
class Config
{
private:
    static std::shared_ptr<Config> config_; 
    cv::FileStorage file_;
    
    Config () {} // private constructor makes a singleton
public:
    ~Config();  // close the file when deconstructing 
    
    // set a new config file 
    static void setParameterFile( const std::string& filename ); 
    
    // access the parameter values
    template< typename T >
    static T get( const std::string& key )
    {
        return T( Config::config_->file_[key] );
    }
};
}

#endif // CONFIG_H
