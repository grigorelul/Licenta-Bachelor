
using Models;

using Models;

namespace Services;

public interface IUserService
{
    Task<UserDto> GetUserAsync(Guid id);
    Task<UserDto> CreateUserAsync(UserDto userDto);
    Task<UserDto> UpdateUserAsync(Guid id, UserDto userDto);
    Task DeleteUserAsync(Guid id);
    Task<IEnumerable<UserDto>> GetUsersAsync();
    Task<IEnumerable<AttendanceDto>> GetUserAttendancesAsync(Guid userId);
    Task<AttendanceDto> CreateAttendanceAsync(Guid userId, AttendanceDto attendanceDto);
    Task<AttendanceDto> UpdateAttendanceAsync(Guid userId, Guid attendanceId, AttendanceDto attendanceDto);
    Task DeleteAttendanceAsync(Guid userId, Guid attendanceId);
}