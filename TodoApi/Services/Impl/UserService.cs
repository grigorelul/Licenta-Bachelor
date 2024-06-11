
using Models;
using Repositories;

namespace Services;

public class UserService : IUserService
{ 
    private readonly IUserRepository _userRepository;
    private readonly IAttendanceRepository _attendanceRepository;

    public UserService(IUserRepository userRepository, IAttendanceRepository attendanceRepository)
    {
        _userRepository = userRepository;
        _attendanceRepository = attendanceRepository;
    }

    public async Task<UserDto> GetUserAsync(Guid id)
    {
        var user = await _userRepository.GetUserAsync(id);
        return UserDto.FromUserToUserDto(user);
    }

        
        
}