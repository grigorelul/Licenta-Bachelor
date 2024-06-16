using Microsoft.EntityFrameworkCore;
using Models;
using Repositories;
using Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
//builder.Services.AddDbContext<TodoContext>(opt =>
//opt.UseInMemoryDatabase("TodoList"));


// Adaugă serviciile CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAllOrigins",
        builderCors =>
        {
            builderCors.AllowAnyOrigin()
                .AllowAnyMethod()
                .AllowAnyHeader();
        });
});

builder.Services.AddTransient<IUserRepository, UserRepository>();
builder.Services.AddTransient<IUserService, UserService>();

builder.Services.AddTransient<IManagerRepository, ManagerRepository>();
builder.Services.AddTransient<IManagerService, ManagerService>();

builder.Services.AddTransient<IAttendanceRepository, AttendanceRepository>();
builder.Services.AddTransient<IAttendanceService, AttendanceService>();

// Adaogă serviciul de bază de date
builder.Services.AddDbContext<MyDbContext>();
// Adaugă alte servicii
builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
    app.UseDeveloperExceptionPage();

}

app.UseHttpsRedirection();
app.UseRouting();

// Activează politica CORS
app.UseCors("AllowAllOrigins");

app.UseAuthorization();

app.MapControllers();

app.Run();
