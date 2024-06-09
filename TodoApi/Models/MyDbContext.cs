using Microsoft.EntityFrameworkCore;

namespace TodoApi.Models;

public class MyDbContext : DbContext
{

    public MyDbContext()
    {
    }
    public MyDbContext(DbContextOptions<MyDbContext> options)
        : base(options)
    {
    }


    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer("Server=localhost;Database=Licenta;Trusted_Connection=True;Encrypt=false");
    }

    public DbSet<User> Users { get; set; }
    public DbSet<Attendance> Attendances { get; set; }
    public DbSet<Manager> Managers { get; set; }
}