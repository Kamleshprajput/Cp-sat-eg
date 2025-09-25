from ortools.sat.python import cp_model
import pandas as pd


def solve_timetable_debug(
    rooms,
    batches,
    subjects,
    teachers,
    fixed_classes=None,
    max_classes_per_day=5,
):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    slots = [1, 2, 3, 4, 5]

    model = cp_model.CpModel()
    x = {}

    for subj, info in subjects.items():
        batch = info["batch"]
        for d in days:
            for s in slots:
                for r in rooms:
                    x[(batch, subj, d, s, r)] = model.NewBoolVar(
                        f"x_{batch}_{subj}_{d}_{s}_{r}"
                    )

    slack_vars = []  # collect slack for debugging

    # Subject weekly requirements
    for subj, info in subjects.items():
        batch = info["batch"]
        lhs = sum(x[(batch, subj, d, s, r)] for d in days for s in slots for r in rooms)
        slack = model.NewIntVar(0, 10, f"slack_subject_{subj}_{batch}")
        model.Add(lhs + slack == info["sessions_per_week"])
        slack_vars.append(("Subject", subj, batch, slack))

    # Batch clash
    for batch in batches:
        for d in days:
            for s in slots:
                lhs = sum(
                    x[(batch, subj, d, s, r)]
                    for subj, info in subjects.items()
                    if info["batch"] == batch
                    for r in rooms
                )
                slack = model.NewIntVar(0, 1, f"slack_batch_{batch}_{d}_{s}")
                model.Add(lhs <= 1 + slack)
                slack_vars.append(("BatchClash", batch, f"{d}-{s}", slack))

    # Room clash
    for r in rooms:
        for d in days:
            for s in slots:
                lhs = sum(
                    x[(info["batch"], subj, d, s, r)] for subj, info in subjects.items()
                )
                slack = model.NewIntVar(0, 1, f"slack_room_{r}_{d}_{s}")
                model.Add(lhs <= 1 + slack)
                slack_vars.append(("RoomClash", r, f"{d}-{s}", slack))

    # Teacher weekly load
    for teacher, tinfo in teachers.items():
        lhs = sum(
            x[(info["batch"], subj, d, s, r)]
            for subj, info in subjects.items()
            if info["teacher"] == teacher
            for d in days
            for s in slots
            for r in rooms
        )
        slack = model.NewIntVar(0, 20, f"slack_teacher_week_{teacher}")
        model.Add(lhs <= tinfo["max_load"] + slack)
        slack_vars.append(("TeacherWeek", teacher, "All", slack))

        # Teacher daily load
        for d in days:
            lhs = sum(
                x[(info["batch"], subj, d, s, r)]
                for subj, info in subjects.items()
                if info["teacher"] == teacher
                for s in slots
                for r in rooms
            )
            slack = model.NewIntVar(0, 5, f"slack_teacher_day_{teacher}_{d}")
            model.Add(lhs <= max_classes_per_day + slack)
            slack_vars.append(("TeacherDay", teacher, d, slack))

    # Fixed classes
    if fixed_classes:
        for batch, subj, d, s, r in fixed_classes:
            lhs = x[(batch, subj, d, s, r)]
            slack = model.NewIntVar(0, 1, f"slack_fixed_{batch}_{subj}_{d}_{s}")
            model.Add(lhs + slack == 1)
            slack_vars.append(("FixedClass", batch, f"{subj}-{d}-{s}", slack))

    # Objective: minimize total slack (find minimal constraint violations)
    model.Minimize(sum(s for _, _, _, s in slack_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("❌ Still no solution.")
        return None

    # Report violated constraints
    print("\n🔍 Constraint check:")
    for kind, who, when, slack in slack_vars:
        val = solver.Value(slack)
        if val > 0:
            print(f"⚠️ Violation -> {kind} | {who} | {when} | slack={val}")

    # Build timetable
    timetable = []
    for d in days:
        for s in slots:
            for batch in batches:
                for subj, info in subjects.items():
                    for r in rooms:
                        if solver.Value(x[(batch, subj, d, s, r)]):
                            timetable.append(
                                {
                                    "Day": d,
                                    "Slot": s,
                                    "Batch": batch,
                                    "Subject": subj,
                                    "Teacher": info["teacher"],
                                    "Room": r,
                                }
                            )

    df = pd.DataFrame(timetable)
    return df.sort_values(["Batch", "Day", "Slot"])


if __name__ == "__main__":
    rooms = ["R1"]
    batches = ["B1"]
    subjects = {
        "Math": {"sessions_per_week": 6, "teacher": "T1", "batch": "B1"},
        "Physics": {"sessions_per_week": 6, "teacher": "T1", "batch": "B1"},
    }
    teachers = {"T1": {"max_load": 12}}
    fixed = [("B1", "Math", "Mon", 1, "R1")]

    timetable = solve_timetable_debug(rooms, batches, subjects, teachers, fixed)

    if timetable is not None:
        print(timetable)
