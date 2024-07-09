using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Backend.Migrations
{
    /// <inheritdoc />
    public partial class migrare2 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances");

            migrationBuilder.DropForeignKey(
                name: "FK_Attendances_Users_UserId",
                table: "Attendances");

            migrationBuilder.AlterColumn<Guid>(
                name: "UserId",
                table: "Attendances",
                type: "uniqueidentifier",
                nullable: true,
                oldClrType: typeof(Guid),
                oldType: "uniqueidentifier");

            migrationBuilder.AlterColumn<Guid>(
                name: "ManagerId",
                table: "Attendances",
                type: "uniqueidentifier",
                nullable: true,
                oldClrType: typeof(Guid),
                oldType: "uniqueidentifier");

            migrationBuilder.AddForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances",
                column: "ManagerId",
                principalTable: "Managers",
                principalColumn: "Id");

            migrationBuilder.AddForeignKey(
                name: "FK_Attendances_Users_UserId",
                table: "Attendances",
                column: "UserId",
                principalTable: "Users",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances");

            migrationBuilder.DropForeignKey(
                name: "FK_Attendances_Users_UserId",
                table: "Attendances");

            migrationBuilder.AlterColumn<Guid>(
                name: "UserId",
                table: "Attendances",
                type: "uniqueidentifier",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"),
                oldClrType: typeof(Guid),
                oldType: "uniqueidentifier",
                oldNullable: true);

            migrationBuilder.AlterColumn<Guid>(
                name: "ManagerId",
                table: "Attendances",
                type: "uniqueidentifier",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"),
                oldClrType: typeof(Guid),
                oldType: "uniqueidentifier",
                oldNullable: true);

            migrationBuilder.AddForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances",
                column: "ManagerId",
                principalTable: "Managers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);

            migrationBuilder.AddForeignKey(
                name: "FK_Attendances_Users_UserId",
                table: "Attendances",
                column: "UserId",
                principalTable: "Users",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
