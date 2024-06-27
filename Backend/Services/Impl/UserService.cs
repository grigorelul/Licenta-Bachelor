
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
    
    public async Task<UserDto> CreateUserAsync(UserDto userDto)
    {
        var user = User.FromUserDtoToUser(userDto);
        user = await _userRepository.CreateUserAsync(user);
        return UserDto.FromUserToUserDto(user);
    }
    
    public async Task<UserDto> UpdateUserAsync(Guid id, UserDto userDto)
    {
        var user = User.FromUserDtoToUser(userDto);
        user.Id = id;
        user = await _userRepository.UpdateUserAsync(user);
        return UserDto.FromUserToUserDto(user);
    }
    
    public async Task DeleteUserAsync(Guid id)
    {
        await _userRepository.DeleteUserAsync(id);
    }
    
    public async Task<IEnumerable<UserDto>> GetUsersAsync()
    {
        var users = await _userRepository.GetUsersAsync();
        return users.Select(UserDto.FromUserToUserDto);
    }
    
    public async Task<IEnumerable<AttendanceDto>> GetUserAttendancesAsync(Guid userId)
    {
        var attendances = await _userRepository.GetUserAttendencesAsync(userId);
        return attendances.Select(AttendanceDto.FromAttendanceToAttendanceDto);
    }
    
    public async Task<AttendanceDto> CreateAttendanceAsync(Guid userId, AttendanceDto attendanceDto)
    {
        var attendance = Attendance.FromAttendanceDtoToAttendance(attendanceDto);
        attendance.UserId = userId;
        attendance = await _attendanceRepository.CreateAttendanceAsync(attendance);
        return AttendanceDto.FromAttendanceToAttendanceDto(attendance);
    }
    
    public async Task<AttendanceDto> UpdateAttendanceAsync(Guid userId, Guid attendanceId, AttendanceDto attendanceDto)
    {
        var attendance = Attendance.FromAttendanceDtoToAttendance(attendanceDto);
        attendance.Id = attendanceId;
        attendance.UserId = userId;
        attendance = await _attendanceRepository.UpdateAttendanceAsync(attendance);
        return AttendanceDto.FromAttendanceToAttendanceDto(attendance);
    }
    
    
    

        
        
}