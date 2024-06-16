
namespace Models;

public class AttendanceDto
{
    public Guid Id { get; set; }
    public DateTime DataSosire { get; set; }
    public DateTime? DataPlecare { get; set; }
    
    public static AttendanceDto FromAttendanceToAttendanceDto(Attendance attendance) =>
        new()
        {
            Id = attendance.Id,
            DataSosire = attendance.DataSosire,
            DataPlecare = attendance.DataPlecare
        };
}